"""Causal cocoa comparison. It does not replace the served price or the evening lot.

Training examples at an origin have a target that is already known. The long
run stays out of update_system.bat.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hybrid_features import (
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    fill_missing_ohlcv,
    future_business_date,
    prepare_market_frame,
    prepare_training_frame,
    resolve_feature_cols,
    trading_sessions,
)
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.served_forecast import direct_level
from src.validation.causal_lot import (
    OUT_OF_SAMPLE_NOTE,
    known_targets,
    matured_origins,
    next_session_present,
    split_windows,
    walk_days,
)
from src.validation.lot_variants import (
    _choose_k,
    _choose_policy,
    _level_errors,
    _policy_score,
    delay_balances,
)
from src.validation.revision_value import block_mean_interval

OUTPUT = ROOT / "reports" / "revision_stability" / "cocoa" / "causal_lot"
HORIZONS = (1, 7, 14, 30)
DECLARED_GRID = {
    "horizons": list(HORIZONS),
    "minimum_distinct_expiries_for_interval": {"1": 2, "7": 14, "14": 28, "30": 60},
    "note": (
        "La grille large, avec une période réservée finale indépendante des choix, "
        "n'est pas exécutée dans ce passage. Soixante échéances distinctes sont le "
        "minimum technique pour qu'un intervalle J+30 puisse démarrer."
    ),
}


class CacheRejected(RuntimeError):
    """A stored fit does not belong to the study that asked for it."""


def study_identity(
    *,
    snapshot_fingerprint: str,
    calendar: str,
    feature_cols: list[str],
    params: dict,
    code_version: str,
    origins: list[str],
) -> dict:
    """One immutable identity. A changed input is a different study."""
    body = {
        "snapshot_fingerprint": snapshot_fingerprint,
        "calendar": calendar,
        "feature_cols": list(feature_cols),
        "params": params,
        "code_version": code_version,
        "origins": list(origins),
    }
    body["study_id"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return body


def history_fingerprint(history: pd.DataFrame) -> str:
    from src.models.hybrid_features import consumed_snapshot_rows
    from src.models.release_manifest import data_fingerprint

    return data_fingerprint(consumed_snapshot_rows(history))


def accept_cached_fit(
    payload: dict,
    *,
    identity: dict,
    history_fingerprint_value: str,
    session: str,
    feature_cols: list[str],
) -> dict:
    """Refuse a fit whose stored inputs are not the ones requested."""
    if not isinstance(payload, dict) or "identity" not in payload:
        raise CacheRejected("Cache sans identité")
    if payload.get("identity") != identity:
        raise CacheRejected("Identité différente")
    if list(payload.get("feature_cols") or []) != list(feature_cols):
        raise CacheRejected("Variables différentes")
    if payload.get("history_fingerprint") != history_fingerprint_value:
        raise CacheRejected("Données différentes")
    if str(payload.get("train_end") or "")[:10] != str(session)[:10]:
        raise CacheRejected("Borne d'entraînement différente")
    learned = payload.get("latest_training_target")
    if learned and str(learned)[:10] > str(session)[:10]:
        raise CacheRejected("Cible postérieure à l'origine")
    return payload


def _frame() -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    raw = prepare_market_frame(
        pd.read_csv(ROOT / "data" / "freeze" / "cocoa_prices.csv"),
        market_id="cocoa",
    )
    frame = fill_missing_ohlcv(trading_sessions(raw))
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    dates = [day.date().isoformat() for day in frame["date"]]
    lookup = {day: float(price) for day, price in zip(dates, frame["price"])}
    return frame, lookup, dates


def _history(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    stamp = pd.Timestamp(session)
    return frame.loc[frame["date"] <= stamp].copy()


def _cache_path(folder: Path, session: str, horizon: int) -> Path:
    return folder / "fits" / f"{session}_h{horizon}.pkl"


def _fit_session(
    history: pd.DataFrame,
    session: str,
    horizons: list[int],
    feature_cols: list[str],
    *,
    folder: Path,
    identity: dict,
) -> None:
    token = history_fingerprint(history)
    dates = [day.date().isoformat() for day in pd.to_datetime(history["date"])]
    missing = []
    for horizon in horizons:
        path = _cache_path(folder, session, horizon)
        if not path.exists():
            missing.append(horizon)
            continue
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        accept_cached_fit(
            payload,
            identity=identity,
            history_fingerprint_value=token,
            session=session,
            feature_cols=feature_cols,
        )
    if not missing:
        return
    features, prophet = prepare_training_frame(history)
    trainer = DirectHorizonTrainer(
        horizons=missing,
        feature_cols=feature_cols,
        xgb_params={**DEFAULT_XGB_PARAMS, "n_jobs": -1},
    )
    models, meta = trainer.fit(history, prophet_model=prophet)
    for column in feature_cols:
        if column not in features.columns:
            features[column] = 0.0
    micro = [column for column in feature_cols if column not in FEATURE_COLS]
    if micro:
        features[micro] = features[micro].ffill().fillna(0.0)
    last = features.dropna(subset=["price"]).iloc[-1]
    for horizon in missing:
        learned = known_targets(dates, session, horizon)
        payload = {
            "model": models[horizon],
            "samples": int(meta["horizons"][str(horizon)]["n_samples"]),
            "feature_cols": list(feature_cols),
            "last_row": last,
            "identity": identity,
            "history_fingerprint": token,
            "train_end": session,
            "latest_training_target": None if not learned else max(learned),
            "training_rows": int(meta["horizons"][str(horizon)]["n_samples"]),
        }
        path = _cache_path(folder, session, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
    print(f"[OK] modèles {session} horizons {missing}", flush=True)


def _predict(
    folder: Path,
    session: str,
    horizon: int,
    spot: float,
    *,
    identity: dict,
    history: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, int, dict]:
    path = _cache_path(folder, session, horizon)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    accept_cached_fit(
        payload,
        identity=identity,
        history_fingerprint_value=history_fingerprint(history),
        session=session,
        feature_cols=feature_cols,
    )
    price = direct_level(
        payload["model"],
        payload["last_row"],
        payload["feature_cols"],
        spot,
        "close_delta",
    )
    bounds = {
        "train_end": payload["train_end"],
        "latest_training_target": payload.get("latest_training_target"),
        "training_rows": int(payload["training_rows"]),
    }
    return price, int(payload["samples"]), bounds


def _prepare_sessions(
    frame: pd.DataFrame,
    lookup: dict[str, float],
    jobs: list[tuple[str, int]],
    feature_cols: list[str],
    *,
    folder: Path,
    identity: dict,
) -> None:
    needed: dict[str, set[int]] = {}
    for origin, horizon in jobs:
        for session, remaining, _target in walk_days(origin, horizon):
            if session not in lookup:
                continue
            needed.setdefault(session, set()).add(remaining)
    for session, horizons in sorted(needed.items()):
        _fit_session(
            _history(frame, session),
            session,
            sorted(horizons),
            feature_cols,
            folder=folder,
            identity=identity,
        )


def _emit_chain(
    frame: pd.DataFrame,
    lookup: dict[str, float],
    dates: list[str],
    origin: str,
    horizon: int,
    feature_cols: list[str],
    *,
    folder: Path,
    identity: dict,
) -> tuple[list[dict], int]:
    del dates
    rows = []
    missing = 0
    reference = None
    for session, remaining, target in walk_days(origin, horizon):
        if session not in lookup:
            missing += 1
            continue
        spot = float(lookup[session])
        if reference is None:
            reference = spot
        price, samples, bounds = _predict(
            folder,
            session,
            remaining,
            spot,
            identity=identity,
            history=_history(frame, session),
            feature_cols=feature_cols,
        )
        learned = bounds.get("latest_training_target")
        if learned and str(learned)[:10] > str(bounds["train_end"])[:10]:
            raise CacheRejected(f"Cible future utilisée le {session}")
        rows.append(
            {
                "origin": origin,
                "session": session,
                "horizon": remaining,
                "initial_horizon": horizon,
                "target": target,
                "role": "initial" if session == origin else "revision",
                "price": price,
                "spot": spot,
                "reference": reference,
                "realized": lookup.get(target),
                "train_end": bounds["train_end"],
                "latest_training_target": learned,
                "training_rows": samples,
            }
        )
    return rows, missing


def _summarize(rows: list[dict], *, k: float | None, policy: str | None, horizon: int) -> dict:
    usable = [row for row in rows if row.get("realized") is not None and row.get("price") is not None]
    openings = [row for row in usable if row["role"] == "initial"]
    by_target: dict[str, list[dict]] = {}
    for row in usable:
        by_target.setdefault(row["target"], []).append(row)
    stable = []
    without_next = []
    for chain in by_target.values():
        if next_session_present(chain):
            stable.extend(chain)
        else:
            without_next.append(chain[0]["target"])
    gains = []
    for row in openings:
        if row.get("spot") is None:
            continue
        gains.append(abs(float(row["spot"]) - float(row["realized"])) - abs(float(row["price"]) - float(row["realized"])))
    low, high = block_mean_interval(gains, horizon, draws=200)
    insufficient = low is None or high is None
    delays = delay_balances(usable)
    established = any(item.get("advantage_established") for item in delays.values())
    return {
        "distinct_expiries": len({row["target"] for row in openings}),
        "without_next_revision": without_next,
        "levels": _level_errors(openings, k),
        "policies": [_policy_score(stable, name) for name in ("initial", "previous", "latest")],
        "chosen_policy_on_this_slice": None if insufficient else policy,
        "sample_status": "échantillon insuffisant" if insufficient else "intervalle disponible",
        "advantage_established": bool(established) and not insufficient,
        "by_remaining_delay": delays,
        "gain_vs_unchanged_low": low,
        "gain_vs_unchanged_high": high,
        "stability_uses_only_chains_with_the_next_session": True,
    }


def _score(rows: list[dict], calibration_targets: set[str], reserved_targets: set[str]) -> dict:
    report = {}
    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if int(row["initial_horizon"]) == horizon]
        calibration = [row for row in horizon_rows if row["origin"] in calibration_targets]
        reserved = [row for row in horizon_rows if row["origin"] in reserved_targets]
        cal_open = [row for row in calibration if row["role"] == "initial"]
        k = _choose_k(cal_open)
        stable = []
        grouped: dict[str, list[dict]] = {}
        for row in calibration:
            grouped.setdefault(row["target"], []).append(row)
        for chain in grouped.values():
            if next_session_present(chain):
                stable.extend(chain)
        policy = _choose_policy(stable)
        calibration_summary = _summarize(calibration, k=k, policy=policy, horizon=horizon)
        reserved_summary = _summarize(reserved, k=k, policy=policy, horizon=horizon)
        insufficient = calibration_summary["sample_status"] == "échantillon insuffisant"
        report[str(horizon)] = {
            "k": k,
            "revision_policy": None if insufficient else policy,
            "sample_status": calibration_summary["sample_status"],
            "advantage_established": reserved_summary["advantage_established"],
            "chosen_on": "calibration",
            "calibration": calibration_summary,
            "reserved": reserved_summary,
        }
    return report


def code_version() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/score_causal_cocoa.py",
        "src/models/direct_horizon_trainer.py",
        "src/models/hybrid_features.py",
    ):
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:16]


def _ensure_study(identity: dict, snapshot: Path) -> Path:
    folder = OUTPUT / "studies" / str(identity["study_id"])
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / "identity.json"
    if identity_path.exists():
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored != identity:
            raise CacheRejected("Identité d'étude incompatible")
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        target = folder / "snapshot.csv"
        if snapshot.exists() and not target.exists():
            shutil.copyfile(snapshot, target)
    return folder


def _write_new_report(folder: Path, mode: str, report: dict) -> Path:
    path = folder / f"{mode}.json"
    if path.exists():
        print(f"[OK] rapport déjà écrit {path}")
        return path
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {path}")
    return path


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "small"
    if mode not in {"small", "extended"}:
        print("[ERREUR] mode small ou extended")
        return 1
    snapshot = ROOT / "data" / "freeze" / "cocoa_prices.csv"
    frame, lookup, dates = _frame()
    feature_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    params = {**DEFAULT_XGB_PARAMS, "n_jobs": -1}
    rows: list[dict] = []
    missing = 0
    if mode == "small":
        origins = [matured_origins(dates, lookup, 14)[-1]]
        identity = study_identity(
            snapshot_fingerprint=history_fingerprint(frame),
            calendar="trading_sessions",
            feature_cols=feature_cols,
            params=params,
            code_version=code_version(),
            origins=origins,
        )
        folder = _ensure_study(identity, snapshot)
        origin = origins[-1]
        _prepare_sessions(frame, lookup, [(origin, 14)], feature_cols, folder=folder, identity=identity)
        walked, absent = _emit_chain(
            frame, lookup, dates, origin, 14, feature_cols, folder=folder, identity=identity
        )
        rows.extend(walked)
        missing += absent
        report = {
            "market": "cocoa",
            "calendar": "trading_sessions",
            "evaluation_kind": "out_of_sample",
            "scope": "small",
            "validated": False,
            "variant_replaces_served_price": False,
            "served_price_unchanged": True,
            "study_id": identity["study_id"],
            "declared_grid": DECLARED_GRID,
            "note": OUT_OF_SAMPLE_NOTE,
            "origin": origin,
            "missing_sessions": missing,
            "rows": rows,
        }
        for row in report["rows"]:
            if row["latest_training_target"] and row["latest_training_target"] > row["train_end"]:
                print("[ERREUR] Cible d'entraînement postérieure à l'origine")
                return 1
    else:
        windows = split_windows(dates, lookup, horizon=30, calibration=8, reserved=8, gap=30)
        origins = list(windows["calibration"]) + list(windows["reserved"])
        identity = study_identity(
            snapshot_fingerprint=history_fingerprint(frame),
            calendar="trading_sessions",
            feature_cols=feature_cols,
            params=params,
            code_version=code_version(),
            origins=origins,
        )
        folder = _ensure_study(identity, snapshot)
        jobs = []
        for horizon in HORIZONS:
            for origin in origins:
                if matured_origins([origin], lookup, horizon):
                    jobs.append((origin, horizon))
        _prepare_sessions(frame, lookup, jobs, feature_cols, folder=folder, identity=identity)
        for origin, horizon in jobs:
            if matured_origins([origin], lookup, horizon):
                walked, absent = _emit_chain(
                    frame, lookup, dates, origin, horizon, feature_cols, folder=folder, identity=identity
                )
                rows.extend(walked)
                missing += absent
        calibration_origins = set(windows["calibration"])
        reserved_origins = set(windows["reserved"])
        report = {
            "market": "cocoa",
            "calendar": "trading_sessions",
            "evaluation_kind": "out_of_sample",
            "scope": "extended",
            "validated": False,
            "variant_replaces_served_price": False,
            "served_price_unchanged": True,
            "study_id": identity["study_id"],
            "declared_grid": DECLARED_GRID,
            "note": OUT_OF_SAMPLE_NOTE,
            "calibration_origins": windows["calibration"],
            "reserved_origins": windows["reserved"],
            "gap_sessions": 30,
            "last_calibration_target": future_business_date(windows["calibration"][-1], 30).date().isoformat(),
            "reserved_starts": windows["reserved"][0],
            "missing_sessions": missing,
            "horizons": _score(rows, calibration_origins, reserved_origins),
        }
        for row in rows:
            learned = row.get("latest_training_target")
            if learned and learned > row["train_end"]:
                print("[ERREUR] Cible d'entraînement postérieure à l'origine")
                return 1
    _write_new_report(folder, mode, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
