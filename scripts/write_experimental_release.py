"""Point the local experimental manifests at the newest trained lot.

The files stay unvalidated. The API reads them only when
PREDICTION_RELEASE_MODE=experimental.
"""

from __future__ import annotations

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


def _candidate(market: str) -> tuple[str, dict] | None:
    base = ROOT / "models" / "candidates" / market
    if not base.is_dir():
        return None
    for folder in sorted((path for path in base.iterdir() if path.is_dir()), reverse=True):
        stamp = folder.name
        prophet = folder / f"prophet_improved_{stamp}.pkl"
        xgboost = folder / f"xgboost_improved_{stamp}.pkl"
        direct = folder / f"model_info_direct_horizon_{stamp}.json"
        improved = folder / f"model_info_improved_{stamp}.json"
        if prophet.exists() and xgboost.exists() and direct.exists():
            relative = f"models/candidates/{market}/{stamp}"
            artifacts = {
                "prophet": f"{relative}/prophet_improved_{stamp}.pkl",
                "xgboost": f"{relative}/xgboost_improved_{stamp}.pkl",
                "direct_info": f"{relative}/model_info_direct_horizon_{stamp}.json",
            }
            if improved.exists():
                artifacts["improved_info"] = f"{relative}/model_info_improved_{stamp}.json"
            return f"improved_{stamp}", artifacts
    return None


def write_market(market: str, models_dir: Path, weights: str, margins: str) -> None:
    found = _candidate(market)
    if found is None:
        print(f"[AVERTISSEMENT] Pas de candidat complet pour {market}")
        return
    version, artifacts = found
    nhits = _latest_dir(models_dir, "nhits_")
    if nhits is not None:
        artifacts["nhits"] = str(nhits.relative_to(ROOT)).replace("\\", "/")
    artifacts["ensemble_weights"] = weights
    artifacts["conformal_intervals"] = margins
    payload = {
        "market": market,
        "version": version,
        "release_mode": "experimental",
        "validated": False,
        "scored_variant": "no_sentiment",
        "artifacts": artifacts,
        "horizons": {key: {"validated": False} for key in ("1", "7", "14", "30")},
    }
    path = ROOT / "config" / f"experimental_release_{market}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {path.name} -> {version} nhits={artifacts.get('nhits', 'absent')}")


def main() -> int:
    write_market(
        "cocoa",
        ROOT / "models",
        "config/ensemble_weights.json",
        "config/conformal_intervals.json",
    )
    write_market(
        "coffee_robusta",
        ROOT / "models" / "coffee_robusta",
        "config/coffee_robusta/ensemble_weights.json",
        "config/coffee_robusta/conformal_intervals.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
