"""The active release is a named file, never the newest model on disk."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]


def release_path(market_id: str, root: Optional[Path] = None) -> Path:
    base = root or ROOT
    return base / "config" / f"active_release_{market_id}.json"


def experimental_release_path(market_id: str, root: Optional[Path] = None) -> Path:
    base = root or ROOT
    return base / "config" / f"experimental_release_{market_id}.json"


def load_experimental_release(market_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Local evening lot. It is never validated, even if the file says otherwise."""
    path = experimental_release_path(market_id, root)
    if not path.exists():
        raise FileNotFoundError(f"Aucun manifeste expérimental pour {market_id}: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("market") != market_id:
        raise ValueError(f"Le manifeste {path} ne concerne pas {market_id}.")
    payload["release_mode"] = "experimental"
    payload["validated"] = False
    horizons = {}
    for key, row in (payload.get("horizons") or {}).items():
        copied = dict(row or {})
        copied["validated"] = False
        horizons[str(key)] = copied
    for key in ("1", "7", "14", "30"):
        horizons.setdefault(key, {"validated": False})
        horizons[key]["validated"] = False
    payload["horizons"] = horizons
    return payload


def release_mode() -> str:
    """Explicit local switch. Anything other than experimental stays on the active file."""
    mode = str(os.getenv("PREDICTION_RELEASE_MODE") or "active").strip().lower()
    return "experimental" if mode == "experimental" else "active"


def load_serving_release(market_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Experimental mode never falls through to the active manifest."""
    if release_mode() == "experimental":
        return load_experimental_release(market_id, root)
    payload = load_active_release(market_id, root)
    payload["release_mode"] = "active"
    return payload


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
    """Newest complete training directory for this market.

    A folder without the named Prophet and XGBoost files, such as a weather
    study, is skipped so it cannot hide the candidate the manifest would load.
    """
    base = (root or ROOT) / "models" / "candidates" / market_id
    if not base.exists():
        return None
    stamps = sorted(path for path in base.iterdir() if path.is_dir())
    for folder_path in reversed(stamps):
        stamp = folder_path.name
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
            continue
        nhits_dirs = sorted(folder_path.glob("nhits_*"))
        if nhits_dirs:
            present["nhits"] = f"{folder}/{nhits_dirs[-1].name}"
        return {"version": f"improved_{stamp}", "artifacts": present}
    return None


def promote_accepted_horizons(
    market_id: str,
    accepted_horizons: list,
    source_report: str,
    conformal_payload: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
    candidate: Optional[Dict[str, Any]] = None,
) -> bool:
    """Mark only the horizons that passed. An empty list leaves the file untouched.

    Artifacts change only when the caller names the candidate that was evaluated.
    A newer folder on disk is not consulted.
    """
    accepted = {str(horizon) for horizon in accepted_horizons}
    if not accepted:
        return False
    release = load_active_release(market_id, root)
    if candidate is not None:
        named_report = str(candidate.get("source_report") or "")
        if named_report and named_report != str(source_report):
            return False
        for key in ("prophet", "xgboost", "direct_info"):
            if key not in (candidate.get("artifacts") or {}):
                return False
        release["version"] = candidate["version"]
        release["artifacts"] = dict(candidate["artifacts"])
        release["procedure_fingerprint"] = candidate.get("procedure_fingerprint")
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


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def procedure_fingerprint(
    *,
    root: Optional[Path] = None,
    artifact_paths: Optional[Dict[str, str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    frame_token: str = "",
) -> str:
    """Hash the procedure that a measurement actually ran, plus a data token."""
    base = root or ROOT
    digest = hashlib.sha256()
    code_paths = [
        "src/models/hybrid_features.py",
        "src/models/weather_features.py",
        "src/models/served_forecast.py",
        "src/validation/walk_forward_validator.py",
        "src/validation/acceptance.py",
        "src/models/nhits_replay.py",
        "config/acceptance.json",
        "config/exchange_holidays.json",
    ]
    for relative in code_paths:
        digest.update(relative.encode("utf-8"))
        digest.update(_file_digest(base / relative).encode("utf-8"))
    for key in sorted((artifact_paths or {})):
        digest.update(key.encode("utf-8"))
        raw = str((artifact_paths or {})[key])
        path = Path(raw)
        if not path.is_absolute():
            path = base / path
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    digest.update(str(child.relative_to(base)).encode("utf-8"))
                    digest.update(_file_digest(child).encode("utf-8"))
        else:
            digest.update(_file_digest(path).encode("utf-8"))
    digest.update(json.dumps(parameters or {}, sort_keys=True, default=str).encode("utf-8"))
    digest.update(str(frame_token).encode("utf-8"))
    return digest.hexdigest()


def data_fingerprint(rows: list) -> str:
    """Hash session dates, prices and open interest. A same-day correction changes it."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()
