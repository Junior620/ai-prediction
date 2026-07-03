"""Load latest walk-forward validation reports from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def find_latest_summary(reports_dir: str = "reports/walk_forward") -> Optional[Path]:
    """Return path to the most recent *_summary.json report."""
    root = Path(reports_dir)
    if not root.exists():
        return None
    candidates = sorted(root.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_walk_forward_csv(reports_dir: str = "reports/walk_forward") -> Optional[str]:
    """Return path to the most recent walk-forward predictions CSV."""
    root = Path(reports_dir)
    if not root.exists():
        return None
    candidates = sorted(
        root.glob("*_walk_forward_predictions.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def load_latest_summary(reports_dir: str = "reports/walk_forward") -> Optional[Dict[str, Any]]:
    """Load the latest walk-forward summary JSON."""
    path = find_latest_summary(reports_dir)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["_report_path"] = str(path)
    return data


def extract_walk_forward_reference(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract compact metrics block for model_info / API."""
    if not summary or "walk_forward" not in summary:
        return None
    wf = summary["walk_forward"]
    xgb = wf.get("summary_by_component", {}).get("xgb_pred", {})
    return {
        "source_report": Path(summary.get("_report_path", "")).stem.replace("_summary", ""),
        "timestamp": summary.get("timestamp"),
        "n_origins": wf.get("n_origins"),
        "horizons": wf.get("horizons"),
        "mape_by_horizon": {
            str(h): metrics.get("mape")
            for h, metrics in xgb.items()
        },
        "rmse_by_horizon": {
            str(h): metrics.get("rmse")
            for h, metrics in xgb.items()
        },
        "directional_accuracy_by_horizon": {
            str(h): metrics.get("directional_accuracy")
            for h, metrics in xgb.items()
        },
        "validation_type": "walk_forward_multi_horizon",
    }
