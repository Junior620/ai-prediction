"""Décide chaque jour s'il faut réentraîner un marché, et écrit pourquoi.

Réentraînement prévu : le samedi, quand le lot en place a plus de 7 jours, ou sur demande.
Deux déclencheurs anticipés s'y ajoutent :

- choc de marché : une séance dont le rendement dépasse 4 fois la médiane des rendements
  absolus des 250 séances précédentes, ou la volatilité sur 20 séances qui franchit 1,5 fois
  celle sur 250, après la date d'entraînement du lot en place (le lot n'a pas vu ce choc) ;
- dégradation : sur les 20 dernières échéances J+1, le rapport entre l'erreur du modèle et
  celle du cours inchangé dépasse 1,2 fois ce rapport sur les 60 précédentes (au moins 15
  échéances de chaque côté, lots de l'inventaire de contamination exclus).

Un changement d'échéance n'est pas un déclencheur : le choc se mesure sur la série sans saut.
Sans série ajustée (robusta avant le lot B), le choc n'est pas évalué plutôt que de prendre
un roulement pour un choc. Un déclencheur anticipé est ignoré si le lot a moins de 3 jours.

Garde : aucun réentraînement quand la séance attendue manque, ni quand la base contient une
séance plus récente que la dernière journée complète. Le lot en place reste alors chargé.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = ROOT / "data" / "status"
CONTAMINATION_PATH = ROOT / "data" / "quarantine" / "oi_contamination.json"

WEEKLY_WEEKDAY = 5  # samedi
MAX_LOT_AGE_DAYS = 7
EARLY_COOLDOWN_DAYS = 3
SHOCK_RETURN_MULTIPLE = 4.0
SHOCK_VOL_RATIO = 1.5
SHORT_WINDOW = 20
LONG_WINDOW = 250
DEGRADATION_RATIO = 1.2
RECENT_MATURED = 20
REFERENCE_MATURED = 60
MIN_MATURED = 15
# Une journée datée D est définitive à partir de D 21:00 UTC (même règle que Databento).
DAY_COMPLETE_HOUR_UTC = 21


@dataclass
class RetrainDecision:
    market: str
    retrain: bool
    wanted: bool
    reasons: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    lot_version: Optional[str] = None
    lot_trained_on: Optional[str] = None
    lot_age_days: Optional[int] = None
    shock: dict[str, Any] = field(default_factory=dict)
    degradation: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    decided_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.retrain:
            return "réentraînement : " + "; ".join(self.reasons)
        if self.wanted:
            return "réentraînement reporté : " + "; ".join(self.blocked_by)
        return "pas de réentraînement : " + ("; ".join(self.ignored) or "lot récent, aucun déclencheur")


def lot_trained_on(version: Optional[str]) -> Optional[date]:
    """Date d'entraînement lue dans le nom du lot (improved_AAAAMMJJ_HHMMSS)."""
    match = re.search(r"(20\d{2})(\d{2})(\d{2})_\d{6}", str(version or ""))
    if not match:
        return None
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def contaminated_versions(path: Optional[Path] = None) -> set[str]:
    """Lots de l'inventaire A0 (faux open interest) : exclus des mesures de performance."""
    target = Path(path) if path is not None else CONTAMINATION_PATH
    if not target.exists():
        return set()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(item) for item in payload.get("contaminated_versions") or []}


def clock_last_complete_day(now: Optional[datetime] = None, is_closed=None) -> str:
    """Dernière séance définitive selon l'horloge, quand aucune plage Databento n'est lue."""
    from src.monitoring.freshness import expected_last_session

    stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    stamp = stamp.tz_convert(timezone.utc).tz_localize(None)
    day = stamp.normalize()
    closed = is_closed
    if closed is None:
        from src.models.hybrid_features import is_closed_collection_date as closed
    if stamp.hour >= DAY_COMPLETE_HOUR_UTC and not closed(day):
        return day.strftime("%Y-%m-%d")
    return expected_last_session(day, closed).strftime("%Y-%m-%d")


def training_guard(
    freshness: Optional[Mapping[str, Any]],
    raw_max_date: Optional[str],
    last_complete_day: Optional[str],
) -> dict[str, Any]:
    """Peut-on entraîner sur les données figées ? Jamais sur une séance incomplète."""
    reasons = []
    status = (freshness or {}).get("status")
    if status != "ok":
        detail = (freshness or {}).get("reason") or "rapport de fraîcheur absent"
        reasons.append(f"séance attendue manquante ({detail})")
    if not last_complete_day:
        reasons.append("dernière journée complète inconnue")
    elif raw_max_date and str(raw_max_date)[:10] > str(last_complete_day)[:10]:
        reasons.append(
            f"séance {str(raw_max_date)[:10]} incomplète (dernière journée complète {last_complete_day})"
        )
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "freshness_status": status,
        "expected_last_session": (freshness or {}).get("expected_last_session"),
        "raw_max_date": None if raw_max_date is None else str(raw_max_date)[:10],
        "last_complete_day": last_complete_day,
    }


def market_shock(closes: Optional[pd.Series], since: Optional[date]) -> dict[str, Any]:
    """Premier choc postérieur à ``since`` sur une série de clôtures sans saut."""
    if closes is None:
        return {"evaluated": False, "triggered": False, "reason": "pas de série sans saut pour ce marché"}
    series = pd.Series(closes, dtype=float).dropna()
    series.index = pd.to_datetime(series.index).normalize()
    series = series.sort_index()
    returns = np.log(series).diff().dropna()
    if len(returns) <= LONG_WINDOW:
        return {"evaluated": False, "triggered": False, "reason": f"moins de {LONG_WINDOW + 1} séances"}
    abs_ret = returns.abs()
    median_prior = abs_ret.rolling(LONG_WINDOW).median().shift(1)
    vol_short = returns.rolling(SHORT_WINDOW).std()
    vol_long = returns.rolling(LONG_WINDOW).std()
    ratio = vol_short / vol_long
    jump = abs_ret > SHOCK_RETURN_MULTIPLE * median_prior
    high = ratio > SHOCK_VOL_RATIO
    crossing = high & ~high.shift(1, fill_value=False)
    start = pd.Timestamp(since) if since is not None else returns.index[0]
    after = returns.index > start
    jumps = list(returns.index[jump & after])
    crossings = list(returns.index[crossing & after])
    last = returns.index[-1]
    result = {
        "evaluated": True,
        "triggered": bool(jumps or crossings),
        "since": start.strftime("%Y-%m-%d"),
        "last_session": last.strftime("%Y-%m-%d"),
        "last_abs_return": round(float(abs_ret.iloc[-1]), 5),
        "median_abs_return_250": None if pd.isna(median_prior.iloc[-1]) else round(float(median_prior.iloc[-1]), 5),
        "vol20_over_vol250": None if pd.isna(ratio.iloc[-1]) else round(float(ratio.iloc[-1]), 3),
        "return_shocks": [d.strftime("%Y-%m-%d") for d in jumps],
        "volatility_crossings": [d.strftime("%Y-%m-%d") for d in crossings],
    }
    return result


def _matured_one_day(
    rows: Iterable[Mapping[str, Any]],
    actuals: Mapping[str, float],
    market_names: set[str],
    excluded: set[str],
) -> list[dict[str, Any]]:
    """Une ligne J+1 par échéance : la dernière journalisée, avec un prix réel connu."""
    by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("market")) not in market_names or int(row.get("horizon") or 0) != 1:
            continue
        if row.get("diagnostic") or row.get("journal_only"):
            continue
        if str(row.get("model_version")) in excluded:
            continue
        price = row.get("price")
        naive = row.get("naive_price", row.get("origin_price"))
        target = str(row.get("target_date") or "")[:10]
        if price is None or naive is None or not target:
            continue
        actual = actuals.get(target)
        if actual is None:
            continue
        by_target[target] = {
            "target_date": target,
            "model_error": abs(float(price) - float(actual)),
            "naive_error": abs(float(naive) - float(actual)),
            "model_version": row.get("model_version"),
        }
    return [by_target[key] for key in sorted(by_target)]


def _ratio(items: list[dict[str, Any]]) -> Optional[float]:
    naive = sum(item["naive_error"] for item in items)
    if naive <= 0:
        return None
    return sum(item["model_error"] for item in items) / naive


def degradation(
    rows: Iterable[Mapping[str, Any]],
    actuals: Mapping[str, float],
    market_names: set[str],
    excluded: Optional[set[str]] = None,
) -> dict[str, Any]:
    matured = _matured_one_day(rows, actuals, market_names, set(excluded or ()))
    recent = matured[-RECENT_MATURED:]
    reference = matured[:-RECENT_MATURED][-REFERENCE_MATURED:] if len(matured) > RECENT_MATURED else []
    result: dict[str, Any] = {
        "evaluated": False,
        "triggered": False,
        "matured": len(matured),
        "recent_n": len(recent),
        "reference_n": len(reference),
        "excluded_versions": sorted(excluded or ()),
    }
    if len(recent) < MIN_MATURED or len(reference) < MIN_MATURED:
        result["reason"] = f"moins de {MIN_MATURED} échéances J+1 échues de chaque côté"
        return result
    recent_ratio = _ratio(recent)
    reference_ratio = _ratio(reference)
    result.update(
        {
            "evaluated": recent_ratio is not None and reference_ratio is not None,
            "recent_ratio": None if recent_ratio is None else round(recent_ratio, 4),
            "reference_ratio": None if reference_ratio is None else round(reference_ratio, 4),
            "threshold": DEGRADATION_RATIO,
        }
    )
    if recent_ratio is not None and reference_ratio:
        result["triggered"] = recent_ratio > DEGRADATION_RATIO * reference_ratio
    return result


def decide(
    market: str,
    *,
    today: date,
    lot_version: Optional[str],
    guard: Mapping[str, Any],
    force: bool = False,
    weekly: bool = False,
    shock: Optional[Mapping[str, Any]] = None,
    degraded: Optional[Mapping[str, Any]] = None,
) -> RetrainDecision:
    trained = lot_trained_on(lot_version)
    age = None if trained is None else (today - trained).days
    reasons: list[str] = []
    ignored: list[str] = []
    if lot_version is None or trained is None:
        reasons.append("aucun lot en place")
    if force:
        reasons.append("--retrain demandé")
    if weekly:
        reasons.append("mode hebdomadaire")
    if today.weekday() == WEEKLY_WEEKDAY and trained is not None and trained < today:
        reasons.append("samedi")
    if age is not None and age > MAX_LOT_AGE_DAYS:
        reasons.append(f"lot de {age} jours (> {MAX_LOT_AGE_DAYS})")
    early = []
    if shock and shock.get("triggered"):
        dates = list(shock.get("return_shocks") or []) + list(shock.get("volatility_crossings") or [])
        early.append(f"choc de marché ({', '.join(sorted(dates)[-3:])})")
    if degraded and degraded.get("triggered"):
        early.append(
            f"dégradation J+1 ({degraded.get('recent_ratio')} contre {degraded.get('reference_ratio')})"
        )
    if early:
        if age is not None and age < EARLY_COOLDOWN_DAYS and not reasons:
            ignored.extend(f"{item} ignoré : lot de {age} jour(s), délai minimal {EARLY_COOLDOWN_DAYS}" for item in early)
        else:
            reasons.extend(early)
    wanted = bool(reasons)
    allowed = bool(guard.get("allowed"))
    return RetrainDecision(
        market=market,
        retrain=wanted and allowed,
        wanted=wanted,
        reasons=reasons,
        ignored=ignored,
        blocked_by=list(guard.get("reasons") or []) if wanted and not allowed else [],
        lot_version=lot_version,
        lot_trained_on=None if trained is None else trained.isoformat(),
        lot_age_days=age,
        shock=dict(shock or {}),
        degradation=dict(degraded or {}),
        guard=dict(guard),
        decided_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def decision_path(market: str, status_dir: Optional[Path] = None) -> Path:
    return Path(status_dir or STATUS_DIR) / f"retrain_{market}.json"


def write_decision(decision: RetrainDecision, status_dir: Optional[Path] = None) -> Path:
    path = decision_path(decision.market, status_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
