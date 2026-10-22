"""Série robusta Londres sans saut de contrat (ICE Futures Europe, racine ``RC`` dans IFEU.IMPACT).

Même règle que le cacao (``src/data_collection/cocoa_roll.py``) : on tient une échéance
et on ne passe à une plus lointaine que lorsque son volume dépasse celui de l'échéance
tenue, ou quand celle-ci disparaît. Chaque rendement est calculé sur un même contrat ;
le niveau est ajusté par ratio sur la dernière clôture de l'échéance tenue.

Le brut est versionné (``src/data_collection/versioned_store.py``) : la série se
reconstruit telle qu'on la connaissait à une date donnée (``build_series(as_of=...)``).

``RC`` confirmé le 22/10/2026 par les métadonnées gratuites : ``RC.c.0..2`` se résolvent
dans IFEU.IMPACT, sur des contrats ``RC  FM<mois><année>!`` aux mois F, H, K, N, U, X
(janvier, mars, mai, juillet, septembre, novembre : les échéances du robusta).
Coûts ``metadata.get_cost`` (ohlcv-1d, RC.c.0..2) : historique depuis le 01/10/2019
1,226 $ ; une semaine 0,004 $.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data_collection.cocoa_roll import DATASET, fetch_calendar_bars, liquid_roll_series, on_exchange_bars
from src.data_collection.versioned_store import VersionedBarStore, utc_stamp

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "databento_robusta_calendar_raw.csv"
SERIES_PATH = ROOT / "data" / "robusta_roll_series.csv"
CALENDAR_SYMBOLS = ("RC.c.0", "RC.c.1", "RC.c.2")
HISTORY_START = "2019-10-01"
# Accord du propriétaire : environ 1,22 $ pour l'historique (1,226 $ estimés le 22/10/2026).
HISTORY_COST_CAP_USD = 1.23
# Une semaine coûte 0,004 $ : au-delà de ce plafond, quelque chose ne va pas (brut perdu, plage énorme).
DAILY_COST_CAP_USD = 0.05
OVERLAP_DAYS = 7
REFRESH_HINT = "collect_coffee_robusta_price.py"


class CostRefused(RuntimeError):
    """Le coût estimé dépasse le plafond : rien n'est demandé."""

    def __init__(self, cost: float, cap: float, start: str, end: str):
        self.cost, self.cap, self.start, self.end = cost, cap, start, end
        super().__init__(f"coût estimé {cost:.4f} $ > plafond {cap:.2f} $ ({start} -> {end})")


def estimate_cost(client, start: str, end: str) -> float:
    """``metadata.get_cost`` : gratuit, à appeler avant toute requête de données."""
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            symbols=list(CALENDAR_SYMBOLS),
            schema="ohlcv-1d",
            stype_in="continuous",
            start=start,
            end=end,
        )
    )


def incremental_start(store: VersionedBarStore) -> Optional[str]:
    """Une semaine avant la dernière barre connue (révisions comprises) ; None sans historique."""
    last = store.last_event()
    if last is None:
        return None
    return (last.tz_convert(None).normalize() - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")


def fetch_into_store(
    store: VersionedBarStore,
    client,
    start: str,
    end: str,
    *,
    cap: float,
    bounds: Optional[dict[str, Any]] = None,
    kind: str = "daily",
) -> dict[str, Any]:
    """Estime, refuse au-delà de ``cap``, télécharge, puis ajoute les versions nouvelles."""
    cost = estimate_cost(client, start, end)
    if cost > cap:
        raise CostRefused(cost, cap, start, end)
    fresh = fetch_calendar_bars(start, end, client=client, symbols=CALENDAR_SYMBOLS)
    fetched_at = utc_stamp(datetime.now(timezone.utc))
    added = store.append(fresh, fetched_at=fetched_at)
    entry = {
        "kind": kind,
        "dataset": DATASET,
        "symbols": list(CALENDAR_SYMBOLS),
        "schema": "ohlcv-1d",
        "start": start,
        "end_exclusive": end,
        "available_end": (bounds or {}).get("available_end"),
        "last_complete_day": (bounds or {}).get("last_complete_day"),
        "fetched_at": fetched_at,
        "cost_usd": round(cost, 4),
        "rows": int(len(fresh)),
        "new_versions": int(added),
    }
    store.log_fetch(entry)
    return entry


def build_series(store: VersionedBarStore, *, as_of: Any = None, until: Optional[str] = None) -> pd.DataFrame:
    """Série sans saut telle que connue à ``as_of``, jamais au-delà de ``until`` (journée complète)."""
    raw = store.as_known_at(as_of)
    if raw.empty:
        return pd.DataFrame()
    raw = raw[raw["symbol"].astype(str).isin(CALENDAR_SYMBOLS)]
    if until:
        days = pd.to_datetime(raw["ts_event"], utc=True).dt.strftime("%Y-%m-%d")
        raw = raw[days <= str(until)[:10]]
    if raw.empty:
        return pd.DataFrame()
    bars = on_exchange_bars(raw)
    series = liquid_roll_series(bars)
    held = bars[["date", "instrument_id", "rank"]].drop_duplicates(subset=["date", "instrument_id"])
    series = series.merge(held, on=["date", "instrument_id"], how="left")
    series["symbol"] = "RC.c." + series["rank"].astype("Int64").astype(str)
    return series.drop(columns="rank")


def write_series(series: pd.DataFrame, store: VersionedBarStore, path: Path = SERIES_PATH, *, until: Optional[str] = None) -> Path:
    """Écrit la série et, à côté, ce qui l'a produite (versions du brut, borne)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, index=False, date_format="%Y-%m-%d")
    versions = store.versions()
    stamp = {
        "built_at": utc_stamp(datetime.now(timezone.utc)),
        "raw_path": store.path.name,
        "raw_versions": int(len(versions)),
        "raw_last_fetched_at": None if versions.empty else str(versions["fetched_at"].max()),
        "until": until,
        "sessions": int(len(series)),
        "first": None if series.empty else f"{series['date'].min():%Y-%m-%d}",
        "last": None if series.empty else f"{series['date'].max():%Y-%m-%d}",
        "rolls": 0 if series.empty else int(series["roll"].sum()),
    }
    path.with_suffix(".json").write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
