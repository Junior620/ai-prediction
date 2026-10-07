"""The active release is a named file, never the newest model on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]


def release_path(market_id: str, root: Optional[Path] = None) -> Path:
    base = root or ROOT
    return base / "config" / f"active_release_{market_id}.json"


def load_active_release(market_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    path = release_path(market_id, root)
    if not path.exists():
        raise FileNotFoundError(f"Aucun manifeste actif pour {market_id}: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("market") != market_id:
        raise ValueError(f"Le manifeste {path} ne concerne pas {market_id}.")
    return payload


def artifact_path(release: Dict[str, Any], key: str, root: Optional[Path] = None) -> Path:
    relative = release.get("artifacts", {}).get(key)
    if not relative:
        raise KeyError(f"Artefact absent du manifeste: {key}")
    path = Path(relative)
    if not path.is_absolute():
        path = (root or ROOT) / path
    return path


def horizon_is_validated(release: Dict[str, Any], horizon: int) -> bool:
    """A horizon is validated only when this exact release says so."""
    if not release.get("validated"):
        return False
    row = (release.get("horizons") or {}).get(str(horizon)) or {}
    return bool(row.get("validated"))


def report_id(release: Optional[Dict[str, Any]]) -> Optional[str]:
    """Report named by the release. Absent when the manifest names none."""
    if not release:
        return None
    named = release.get("source_report")
    if named:
        return str(named)
    provenance = release.get("margins_provenance") or {}
    named = provenance.get("source_report")
    return str(named) if named else None


def latest_candidate(market_id: str, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Newest training directory for this market, if one exists on disk."""
    base = (root or ROOT) / "models" / "candidates" / market_id
    if not base.exists():
        return None
    stamps = sorted(path for path in base.iterdir() if path.is_dir())
    if not stamps:
        return None
    stamp = stamps[-1].name
    folder = f"models/candidates/{market_id}/{stamp}"
    artifacts = {
        "prophet": f"{folder}/prophet_improved_{stamp}.pkl",
        "xgboost": f"{folder}/xgboost_improved_{stamp}.pkl",
        "direct_info": f"{folder}/model_info_direct_horizon_{stamp}.json",
        "improved_info": f"{folder}/model_info_improved_{stamp}.json",
    }
    present = {
        key: relative
        for key, relative in artifacts.items()
        if ((root or ROOT) / relative).exists()
    }
    if "prophet" not in present or "xgboost" not in present:
        return None
    nhits_dirs = sorted((base / stamp).glob("nhits_*"))
    if nhits_dirs:
        present["nhits"] = f"{folder}/{nhits_dirs[-1].name}"
    return {"version": f"improved_{stamp}", "artifacts": present}


def promote_accepted_horizons(
    market_id: str,
    accepted_horizons: list,
    source_report: str,
    conformal_payload: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> bool:
    """Mark only the horizons that passed. An empty list leaves the file untouched."""
    accepted = {str(horizon) for horizon in accepted_horizons}
    if not accepted:
        return False
    release = load_active_release(market_id, root)
    candidate = latest_candidate(market_id, root)
    if candidate:
        release["version"] = candidate["version"]
        artifacts = dict(release.get("artifacts") or {})
        artifacts.update(candidate["artifacts"])
        release["artifacts"] = artifacts
    horizons = dict(release.get("horizons") or {})
    for key in ("1", "7", "14", "30"):
        row = dict(horizons.get(key) or {})
        row["validated"] = key in accepted
        horizons[key] = row
    release["horizons"] = horizons
    release["validated"] = any(bool(row.get("validated")) for row in horizons.values())
    release["source_report"] = source_report
    release["scored_variant"] = "no_sentiment"
    path = release_path(market_id, root)
    path.write_text(json.dumps(release, indent=2), encoding="utf-8")
    if conformal_payload is not None:
        relative = (release.get("artifacts") or {}).get("conformal_intervals")
        if relative:
            conformal_path = Path(relative)
            if not conformal_path.is_absolute():
                conformal_path = (root or ROOT) / conformal_path
            conformal_path.parent.mkdir(parents=True, exist_ok=True)
            conformal_path.write_text(
                json.dumps(conformal_payload, indent=2),
                encoding="utf-8",
            )
    return True
