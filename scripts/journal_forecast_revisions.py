"""Generate horizons 1–30 for one named lot and journal them once.

The previous lot is scored on the same frozen rows as a diagnostic. That
diagnostic is not a published revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "--market" in sys.argv:
    _market_index = sys.argv.index("--market")
    if _market_index + 1 < len(sys.argv) and sys.argv[_market_index + 1] == "coffee_robusta":
        import neuralforecast  # noqa: F401

import pandas as pd
import yaml

from src.models.horizon_schedule import TRACKED_HORIZONS
from src.models.hybrid_features import clean_price_dataframe
from src.models.market_registry import get_market_config
from src.models.prediction_journal import JournalLockError
from src.models.revision_command import (
    frame_fingerprint,
    journal_served,
    load_named_predictor,
    procedure_for,
)
from scripts.write_experimental_release import named_candidate

def _read_release(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _release_from_stamp(market: str, stamp: str, weights: str, margins: str) -> dict:
    found = named_candidate(market, stamp)
    if found is None:
        raise SystemExit(f"[ERREUR] Lot nommé incomplet: {market} {stamp}")
    version, artifacts, trained = found
    artifacts = dict(artifacts)
    artifacts["ensemble_weights"] = weights
    artifacts["conformal_intervals"] = margins
    from scripts.write_experimental_release import _nhits_required

    if _nhits_required(weights):
        pointer = ROOT / "models" / ("coffee_robusta" if market == "coffee_robusta" else "") / "NHITS_LOT.json"
        if market == "cocoa":
            pointer = ROOT / "models" / "NHITS_LOT.json"
        if pointer.exists():
            relative = str(json.loads(pointer.read_text(encoding="utf-8")).get("relative") or "").replace("\\", "/")
            if relative:
                artifacts["nhits"] = relative
    return {
        "market": market,
        "version": version,
        "release_mode": "experimental",
        "validated": False,
        "artifacts": artifacts,
        "direct_horizons": trained,
        "horizons": {key: {"validated": False} for key in ("1", "7", "14", "30")},
    }


def _served_price(item) -> float | None:
    status = str(getattr(item, "status", None) or (item.components or {}).get("status") or "")
    if status == "unavailable" or item.price is None:
        return None
    return float(item.price)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=("cocoa", "coffee_robusta"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--stamp", default="")
    args = parser.parse_args()
    pointer = ROOT / "models" / "candidates" / args.market / "TRAINED_LOT.json"
    stamp = args.stamp
    if not stamp and pointer.exists():
        stamp = str(json.loads(pointer.read_text(encoding="utf-8")).get("stamp") or "")
    if not stamp:
        current = ROOT / "config" / f"experimental_release_{args.market}.json"
        payload = _read_release(current) or {}
        print(f"[ERREUR] Pas de nouveau lot nommé. Lot encore chargé: {payload.get('version')}")
        return 1

    weights = (
        "config/ensemble_weights.json"
        if args.market == "cocoa"
        else "config/coffee_robusta/ensemble_weights.json"
    )
    margins = (
        "config/conformal_intervals.json"
        if args.market == "cocoa"
        else "config/coffee_robusta/conformal_intervals.json"
    )
    release = _release_from_stamp(args.market, stamp, weights, margins)
    market_cfg = get_market_config(args.market)
    api_market = market_cfg.api_markets[0]
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as handle:
        pred_cfg = (yaml.safe_load(handle) or {}).get("prediction") or {}
    frame = clean_price_dataframe(pd.read_csv(args.snapshot))
    if frame.empty:
        print("[ERREUR] Instantané vide")
        return 1
    token = frame_fingerprint(frame)
    procedure = procedure_for(release, pred_cfg.get("max_abs_change_pct") or {})
    predictor = load_named_predictor(release, market_cfg, pred_cfg)
    predictor.bind_frame(frame)
    predictor.release_mode = "experimental"
    predictor.active_release = release
    try:
        fresh = predictor.predict(list(TRACKED_HORIZONS))
        journal_served(
            fresh,
            market=api_market,
            release=release,
            procedure=procedure,
            data_token=token,
        )
    except JournalLockError as exc:
        print(f"[ERREUR] {exc}")
        return 1
    print(f"[OK] Prévisions {api_market} {release['version']} enregistrées")

    current_path = ROOT / "config" / f"experimental_release_{args.market}.json"
    previous_path = ROOT / "config" / f"previous_experimental_release_{args.market}.json"
    current = _read_release(current_path)
    previous = current
    if current and current.get("version") == release["version"]:
        previous = _read_release(previous_path)
    if not previous or previous.get("version") == release["version"]:
        print("[OK] Pas de lot précédent distinct à comparer")
        return 0
    try:
        old_predictor = load_named_predictor(previous, market_cfg, pred_cfg)
        old_predictor.bind_frame(frame)
        old_predictor.release_mode = "experimental"
        old_predictor.active_release = previous
        old = old_predictor.predict(list(TRACKED_HORIZONS))
        from src.models.served_forecast import publish_or_close

        publish_or_close(old, previous, None, release_mode="experimental")
        previous_prices = {int(item.horizon): _served_price(item) for item in old}
        journal_served(
            fresh,
            market=api_market,
            release=release,
            procedure=procedure,
            data_token=token,
            diagnostic=True,
            previous_version=str(previous.get("version")),
            previous_prices=previous_prices,
        )
    except JournalLockError as exc:
        print(f"[ERREUR] {exc}")
        return 1
    except Exception as exc:
        print(f"[AVERTISSEMENT] Comparaison ancien lot indisponible: {exc}")
        return 0
    print(f"[OK] Diagnostic ancien lot {previous.get('version')} enregistré, hors révisions publiées")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
