"""
Fill prediction accuracy: join matured predictions with spot prices,
compute metrics, store in model_metrics.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from config.settings import get_settings
from src.monitoring.performance_monitor import PerformanceMonitor


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    print("=" * 80)
    print("EVALUATION PREDICTIONS -> PRIX REELS")
    print("=" * 80)

    settings = get_settings()
    sb = create_client(settings.supabase_url, settings.supabase_key)
    monitor = PerformanceMonitor(supabase_client=sb)

    # Predictions whose target date is in the past (up to 90 days history)
    since = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    preds = (
        sb.table("predictions")
        .select("*")
        .gte("created_at", since)
        .order("created_at", desc=False)
        .limit(2000)
        .execute()
    )
    rows = preds.data or []
    print(f"Predictions candidates: {len(rows)}")

    # Spot prices (cocoa ICE_NY / default market in price_data)
    prices = (
        sb.table("price_data")
        .select("timestamp,price,market")
        .gte("timestamp", since)
        .order("timestamp", desc=False)
        .limit(5000)
        .execute()
    )
    price_rows = prices.data or []
    # Index by date (UTC day)
    by_day: dict[str, float] = {}
    for pr in price_rows:
        market = (pr.get("market") or "ICE_NY").upper()
        if market not in ("ICE_NY", "CC", "COCOA", ""):
            # Prefer cocoa for default metrics; skip robusta in this pass
            if "ROBUSTA" in market or "COFFEE" in market:
                continue
        day = _parse_ts(pr["timestamp"]).date().isoformat()
        by_day[day] = float(pr["price"])

    print(f"Prix spot indexes: {len(by_day)} jours")

    y_true, y_pred, y_lo, y_hi = [], [], [], []
    matched = 0
    now = datetime.now(timezone.utc)

    for row in rows:
        created = _parse_ts(row["created_at"])
        horizon = int(row["horizon"])
        target = (created + timedelta(days=horizon)).date()
        if target > now.date():
            continue  # not matured
        actual = by_day.get(target.isoformat())
        if actual is None:
            continue
        matched += 1
        y_true.append(actual)
        y_pred.append(float(row["predicted_price"]))
        y_lo.append(float(row["lower_bound"]))
        y_hi.append(float(row["upper_bound"]))

    print(f"Paires pred/actual matures: {matched}")
    if matched < 3:
        print("[AVERTISSEMENT] Pas assez de paires pour calculer des metriques")
        return 0

    metrics = monitor.compute_metrics(
        np.array(y_true, dtype=float),
        np.array(y_pred, dtype=float),
        np.array(y_lo, dtype=float),
        np.array(y_hi, dtype=float),
    )
    version = "hybrid_daily"
    if rows:
        version = str(rows[-1].get("model_version") or version)

    insert = {
        "model_version": version,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "mape": float(metrics["mape"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
        "coverage_rate": float(metrics["coverage_rate"]),
        "mean_interval_width": float(metrics.get("mean_interval_width") or 0),
    }
    sb.table("model_metrics").insert(insert).execute()

    print("\nMetriques enregistrees:")
    for k, v in insert.items():
        print(f"  {k}: {v}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
