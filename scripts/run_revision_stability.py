"""Replay two cadences on one frozen snapshot. No cadence is chosen."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "--market" in sys.argv:
    _market_index = sys.argv.index("--market")
    if _market_index + 1 < len(sys.argv) and sys.argv[_market_index + 1] == "coffee_robusta":
        import neuralforecast  # noqa: F401

import pandas as pd
import yaml

from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.horizon_schedule import OFFICIAL_HORIZONS, TRACKED_HORIZONS, scheduled_cap, scheduled_weights
from src.models.hybrid_features import (
    FEATURE_COLS,
    OHLCV_FILL,
    build_prediction_row,
    consumed_snapshot_rows,
    fill_missing_ohlcv,
    future_business_date,
    prepare_market_frame,
    resolve_feature_cols,
    session_anomalies,
    trading_sessions,
)
from src.models.hybrid_trainer import HybridModelTrainer
from src.models.market_registry import get_market_config
from src.models.release_manifest import data_fingerprint, procedure_fingerprint
from src.models.served_forecast import compose_served_price, direct_level
from src.validation.revision_stability import (
    ArchivePreserved,
    ContradictorySession,
    annotate_realized,
    append_unique_session,
    load_unique_sessions,
    lot_index,
    replay_policies,
    score_policies,
    study_window,
)


def _last_row(frame: pd.DataFrame, prophet_model, feature_cols: list[str]):
    from src.models.hybrid_features import prepare_training_frame

    built, _ = prepare_training_frame(fill_missing_ohlcv(frame), prophet_model=prophet_model)
    for column in feature_cols:
        if column not in built.columns:
            built[column] = 0.0
    built[feature_cols] = built[feature_cols].ffill()
    clean = built.dropna(subset=["price", "price_lag_30"]).copy()
    if clean.empty:
        return None
    clean[feature_cols] = clean[feature_cols].fillna(0.0)
    return clean.iloc[-1]


def _train(frame: pd.DataFrame, feature_cols: list[str], replay_nhits: bool, unique_id: str):
    filled = fill_missing_ohlcv(frame)
    trainer = HybridModelTrainer(feature_cols=feature_cols)
    prophet, _xgb, _features = trainer.fit(filled)
    direct = DirectHorizonTrainer(horizons=list(TRACKED_HORIZONS), feature_cols=feature_cols)
    models, _meta = direct.fit(filled, prophet_model=prophet)
    network = None
    if replay_nhits:
        from src.models.nhits_replay import fit_nhits_network

        network = fit_nhits_network(filled, unique_id)
    return {
        "prophet": prophet,
        "models": models,
        "target": direct.target,
        "nhits": network,
    }


def _price(bundle, history, origin_price, origin_date, horizon, weights, caps, margins, feature_cols, unique_id):
    """Frozen parameters, features rebuilt on the history of this session."""
    if horizon not in bundle["models"]:
        return None
    last = _last_row(history, bundle["prophet"], feature_cols)
    if last is None:
        return None
    xgb_price = direct_level(
        bundle["models"][horizon],
        last,
        feature_cols,
        origin_price,
        bundle["target"],
    )
    future = future_business_date(origin_date, horizon)
    prophet_row = build_prediction_row(
        last, origin_price, future, bundle["prophet"], feature_cols=feature_cols
    )
    prophet_price = float(prophet_row["prophet_yhat"].iloc[0])
    nhits_price = None
    if float(weights.get("nhits", 0.0) or 0.0) > 0.0:
        from src.models.nhits_replay import predict_nhits_levels

        nhits_price = predict_nhits_levels(bundle.get("nhits"), history, unique_id).get(horizon)
    served = compose_served_price(
        xgb_price=xgb_price,
        prophet_price=prophet_price,
        nhits_price=nhits_price,
        weights=weights,
        spot=origin_price,
        horizon=horizon,
        max_abs_change_pct={**caps, str(horizon): scheduled_cap(horizon, caps)},
        conformal_margins=margins,
        garch_forecast=None,
        confidence_level=0.9,
        price_volatility=0.0,
    )
    price = served.get("price")
    if served.get("engine_missing") or price is None:
        return None
    try:
        if not (float(price) > 0):
            return None
    except (TypeError, ValueError):
        return None
    return float(price)


def _identity(frame: pd.DataFrame, market: str, as_of: str, n_origins: int, mode: str, weights, caps, origins, missing) -> dict:
    return {
        "market": market,
        "as_of": as_of,
        "n_origins": n_origins,
        "mode": mode,
        "origins": [pd.Timestamp(day).date().isoformat() for day in origins],
        "missing_targets": list(missing),
        "anomalies": session_anomalies(frame),
        "data_fingerprint": data_fingerprint(consumed_snapshot_rows(frame)),
        "procedure_fingerprint": procedure_fingerprint(
            parameters={
                "weights": weights,
                "caps": caps,
                "cadence_block": 5,
                "calendar": "trading_sessions",
                "ohlcv_fill": OHLCV_FILL,
            },
            frame_token="",
        ),
    }


def _render(summary: dict, market: str) -> str:
    lines = [
        f"# Stabilité des révisions — {market}",
        "",
        summary["note"],
        "",
        "Le diagnostic de changement de lot est à part des révisions publiées.",
        "L'erreur du cours inchangé est notée sur les mêmes initiales matures.",
        "",
    ]
    for row in summary["policies"]:
        lines.append(
            f"- {row['policy']} J+{row['initial_horizon']}: "
            f"initiales matures {row['mature_initials']}, "
            f"MAPE modèle {row['initial_mape']}, "
            f"MAPE cours inchangé {row.get('unchanged_origin_mape')}, "
            f"révisions {row['revision_count']}, "
            f"MAPE révisions par horizon restant {row.get('revision_mape_by_remaining')}, "
            f"MAPE cours du jour de révision {row.get('unchanged_revision_day_mape')}, "
            f"amplitude médiane {row['median_abs_revision']}, "
            f"amplitude max {row['max_abs_revision']}, "
            f"inversions {row['inversions']}, "
            f"cibles absentes {row.get('absent_targets')}"
        )
    lines.append("")
    lines.append(f"Diagnostics de lot: {summary['diagnostics']}")
    if summary.get("anomalies"):
        lines.append("")
        lines.append("Dates présentes alors que le calendrier les déclare fermées :")
        lines.append(", ".join(summary["anomalies"]))
    if summary.get("missing_targets"):
        lines.append("")
        lines.append(f"Cibles absentes signalées : {len(summary['missing_targets'])}.")
    lines.append("")
    lines.append("Aucune cadence n'est retenue.")
    return "\n".join(lines) + "\n"


def _rebuild_unique(args) -> int:
    """Score the unique sessions already stored. The archive file is not rewritten."""
    source = Path(args.output_dir) / "sessions.jsonl"
    identity_path = Path(args.output_dir) / "checkpoint_identity.json"
    study_id = "archived"
    if identity_path.exists():
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        study_id = str(stored.get("procedure_fingerprint") or study_id)
    batches = load_unique_sessions(source, study_id=study_id)
    records = []
    for batch in batches:
        records.extend(batch.get("rows") or [])
    frame = prepare_market_frame(pd.read_csv(args.snapshot), market_id=args.market)
    frame = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(args.as_of)].copy()
    trading = trading_sessions(frame)
    lookup = {
        pd.Timestamp(row.date).normalize(): float(row.price)
        for row in trading.itertuples(index=False)
    }
    scored = annotate_realized(records, lookup, args.as_of)
    stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "policies": score_policies(scored),
        "diagnostics": sum(1 for row in records if row.get("kind") == "diagnostic"),
        "note": (
            "Synthèse recalculée sur les séances uniques. Le fichier de séances "
            "d'origine n'est pas réécrit. Aucune cadence n'est choisie."
        ),
        "market": args.market,
        "as_of": args.as_of,
        "unique_sessions": len(batches),
        "stored_rows": len(records),
    }
    output = Path(args.output_dir)
    path = output / f"{stamped}_unique_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Synthèse unique {path} séances={len(batches)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=("cocoa", "coffee_robusta"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--as-of", default="2026-10-06")
    parser.add_argument("--origins", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=0, help="Essai court: dernières séances seulement")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rebuild-unique", action="store_true")
    args = parser.parse_args()
    if args.rebuild_unique:
        return _rebuild_unique(args)
    market = get_market_config(args.market)
    raw = prepare_market_frame(pd.read_csv(args.snapshot), market_id=args.market)
    raw = raw[pd.to_datetime(raw["date"]) <= pd.Timestamp(args.as_of)].copy()
    trading = trading_sessions(raw)
    feature_cols = (
        resolve_feature_cols(include_ohlcv=True, include_oi=True)
        if args.market == "cocoa"
        else list(FEATURE_COLS)
    )
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as handle:
        caps = ((yaml.safe_load(handle) or {}).get("prediction") or {}).get("max_abs_change_pct") or {}
    weights_path = Path(market.ensemble_weights_file)
    weights = json.loads(weights_path.read_text(encoding="utf-8")).get("by_horizon") or {}
    margins_path = Path(market.conformal_intervals_file)
    margins = json.loads(margins_path.read_text(encoding="utf-8")) if margins_path.exists() else {}
    replay_nhits = any(float((item or {}).get("nhits") or 0) > 0 for item in weights.values())
    mode = "trial" if args.sessions else "study"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.sessions:
        dates = list(pd.to_datetime(trading["date"]).dt.normalize().drop_duplicates().sort_values())
        sessions = dates[-int(args.sessions) :]
        chosen = list(sessions)
        missing: list[str] = []
    else:
        chosen, sessions, missing = study_window(raw, as_of=args.as_of, n_origins=args.origins)
        if not sessions:
            print("[ERREUR] Aucune séance de cotation dans la fenêtre.")
            return 1
    identity = _identity(raw, args.market, args.as_of, args.origins, mode, weights, caps, chosen, missing)
    identity_path = output / "checkpoint_identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            print("[ERREUR] Point de reprise refusé: l'empreinte a changé.")
            return 2
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    price_lookup = {
        pd.Timestamp(row.date).normalize(): float(row.price)
        for row in trading.itertuples(index=False)
    }
    done_path = output / "sessions.jsonl"
    study_id = identity["procedure_fingerprint"]
    done = set()
    prior = []
    if done_path.exists():
        for batch in load_unique_sessions(done_path, study_id=study_id):
            done.add(batch["session"])
            prior.extend(batch["rows"])

    def history_for(session):
        return trading[pd.to_datetime(trading["date"]) <= pd.Timestamp(session)].copy()

    def train_fn(history):
        return _train(history, feature_cols, replay_nhits, market.nhits_unique_id)

    def price_fn(bundle, history, origin_price, session, horizon):
        return _price(
            bundle,
            history,
            origin_price,
            session,
            horizon,
            scheduled_weights(horizon, weights),
            caps,
            margins,
            feature_cols,
            market.nhits_unique_id,
        )

    origin_dates = {pd.Timestamp(day).normalize() for day in chosen}
    stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
    def _store(batch: dict) -> None:
        state = append_unique_session(done_path, batch, study_id=study_id)
        if state == "duplicate":
            return
        print(
            f"[OK] {batch['session']} lot cinq séances={lot_index(batch['index'])} nhits={replay_nhits}",
            flush=True,
        )

    try:
        batches = replay_policies(
            sessions,
            history_for,
            train_fn,
            price_fn,
            official_horizons=tuple(OFFICIAL_HORIZONS),
            origin_dates=origin_dates,
            done_sessions=done,
            prior_records=prior,
            on_session=_store,
        )
    except (ContradictorySession, ArchivePreserved) as exc:
        print(f"[ERREUR] {exc}")
        return 2
    records = list(prior)
    for batch in batches:
        records.extend(batch["rows"])
    scored = annotate_realized(records, price_lookup, args.as_of)
    summary = {
        "policies": score_policies(scored),
        "diagnostics": sum(1 for row in records if row.get("kind") == "diagnostic"),
        "note": (
            "Précision et stabilité sont lues ensemble. Une estimation qui change moins "
            "souvent peut rester fausse plus longtemps. Aucune cadence n'est choisie. "
            "Ce n'est pas une preuve de couverture des bandes."
        ),
        "market": args.market,
        "as_of": args.as_of,
        "sessions": [pd.Timestamp(day).date().isoformat() for day in sessions],
        "origins": identity["origins"],
        "missing_targets": missing,
        "anomalies": identity["anomalies"],
        "identity": identity,
    }
    stamp_path = output / f"{stamped}_summary.json"
    stamp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / f"{stamped}_summary.md").write_text(_render(summary, args.market), encoding="utf-8")
    print(f"[OK] Synthèse {stamp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
