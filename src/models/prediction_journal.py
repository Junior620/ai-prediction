"""Local forecast journal. Supabase does not store the candidate bounds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "prediction_journal.jsonl"


def append_forecast(
    *,
    market: str,
    model_version: str,
    release_mode: str,
    horizon: int,
    price: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    origin_date: Optional[str],
    origin_price: Optional[float],
    target_date: Optional[str],
    status: str,
) -> None:
    if status not in {"experimental", "validated", "unavailable"}:
        status = "unavailable"
    if release_mode == "experimental" and status == "validated":
        status = "experimental"
    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "model_version": model_version,
        "release_mode": release_mode,
        "horizon": int(horizon),
        "price": price,
        "lower": lower,
        "upper": upper,
        "origin_date": origin_date,
        "origin_price": origin_price,
        "target_date": target_date,
        "status": status,
    }
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_forecasts() -> list[dict[str, Any]]:
    if not JOURNAL.exists():
        return []
    rows = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
