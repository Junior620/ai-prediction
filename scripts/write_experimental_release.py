"""Point one experimental manifest at the lot this run just trained.

The files stay unvalidated. The API reads them only when
PREDICTION_RELEASE_MODE=experimental. The newest folder is never selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest_dir(base: Path, prefix: str) -> Path | None:
    if not base.is_dir():
        return None
    matches = [path for path in base.glob(f"{prefix}*") if path.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.name)[-1]


def named_candidate(market: str, stamp: str) -> tuple[str, dict, list[int]] | None:
    """Artifacts of one named lot. Incomplete lots are refused."""
    folder = ROOT / "models" / "candidates" / market / stamp
    prophet = folder / f"prophet_improved_{stamp}.pkl"
    xgboost = folder / f"xgboost_improved_{stamp}.pkl"
    direct = folder / f"model_info_direct_horizon_{stamp}.json"
    improved = folder / f"model_info_improved_{stamp}.json"
    if not (prophet.exists() and xgboost.exists() and direct.exists() and improved.exists()):
        return None
    meta = json.loads(direct.read_text(encoding="utf-8"))
    trained = sorted(int(key) for key in (meta.get("horizons") or {}))
    relative = f"models/candidates/{market}/{stamp}"
    artifacts = {
        "prophet": f"{relative}/prophet_improved_{stamp}.pkl",
        "xgboost": f"{relative}/xgboost_improved_{stamp}.pkl",
        "direct_info": f"{relative}/model_info_direct_horizon_{stamp}.json",
        "improved_info": f"{relative}/model_info_improved_{stamp}.json",
    }
    return f"improved_{stamp}", artifacts, trained


def _nhits_required(weights: str) -> bool:
    path = ROOT / weights
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in (payload.get("by_horizon") or {}).values():
        try:
            if float((item or {}).get("nhits") or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _active_nhits(market: str) -> str | None:
    path = ROOT / "config" / f"active_release_{market}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    relative = (payload.get("artifacts") or {}).get("nhits")
    if not relative:
        return None
    return str(relative).replace("\\", "/")


def keep_evening_nhits(relative: str | None, active_relative: str | None) -> str | None:
    """The evening lot does not reuse the network already named by the active release."""
    if not relative:
        return None
    cleaned = relative.replace("\\", "/")
    if active_relative and cleaned == active_relative.replace("\\", "/"):
        return None
    return cleaned


def write_market(market: str, models_dir: Path, weights: str, margins: str, stamp: str) -> int:
    found = named_candidate(market, stamp)
    if found is None:
        print(f"[ERREUR] Lot nommé incomplet pour {market}: {stamp}")
        return 1
    version, artifacts, trained = found
    if _nhits_required(weights):
        pointer = models_dir / "NHITS_LOT.json"
        relative = None
        if pointer.exists():
            relative = str(json.loads(pointer.read_text(encoding="utf-8")).get("relative") or "")
            relative = relative.replace("\\", "/")
        else:
            print(f"[AVERTISSEMENT] Pas de N-HiTS nommé pour {market}")
        kept = keep_evening_nhits(relative, _active_nhits(market))
        if relative and kept is None:
            print(f"[AVERTISSEMENT] N-HiTS {relative} est celui du manifeste actif — non repris")
        if kept:
            artifacts["nhits"] = kept
    artifacts["ensemble_weights"] = weights
    artifacts["conformal_intervals"] = margins
    payload = {
        "market": market,
        "version": version,
        "release_mode": "experimental",
        "validated": False,
        "scored_variant": "no_sentiment",
        "artifacts": artifacts,
        "direct_horizons": trained,
        "horizons": {key: {"validated": False} for key in ("1", "7", "14", "30")},
    }
    path = ROOT / "config" / f"experimental_release_{market}.json"
    previous = ROOT / "config" / f"previous_experimental_release_{market}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("version") != version:
            previous.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[OK] Lot précédent conservé: {old.get('version')}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload.get("validated"):
        print("[ERREUR] Le manifeste expérimental ne peut pas être validé")
        return 1
    print(
        f"[OK] {path.name} -> {version} nhits={artifacts.get('nhits', 'absent')} "
        f"horizons={trained} evaluation_report=absent"
    )
    return 0


def _stamp_from_pointer(market: str, stamp: str) -> str:
    if stamp:
        return stamp
    pointer = ROOT / "models" / "candidates" / market / "TRAINED_LOT.json"
    if not pointer.exists():
        return ""
    return str(json.loads(pointer.read_text(encoding="utf-8")).get("stamp") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=("cocoa", "coffee_robusta"))
    parser.add_argument("--stamp", default="")
    args = parser.parse_args()
    stamp = _stamp_from_pointer(args.market, args.stamp)
    if not stamp:
        print("[ERREUR] Aucun lot nommé. Le dernier dossier n'est pas utilisé.")
        return 1
    if args.market == "cocoa":
        return write_market(
            "cocoa",
            ROOT / "models",
            "config/ensemble_weights.json",
            "config/conformal_intervals.json",
            stamp,
        )
    return write_market(
        "coffee_robusta",
        ROOT / "models" / "coffee_robusta",
        "config/coffee_robusta/ensemble_weights.json",
        "config/coffee_robusta/conformal_intervals.json",
        stamp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
