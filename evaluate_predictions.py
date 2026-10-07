"""
Join matured predictions with the spot of the stored target date and market.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from config.settings import get_settings
from src.models.market_registry import resolve_api_market
from src.monitoring.performance_monitor import PerformanceMonitor


def scored_forecast(row: dict) -> tuple[float, str]:
    """Model price when the journal kept one. Otherwise the published price."""
    candidate = row.get("candidate_price")
    if candidate is not None and candidate != "":
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            return value, "candidat"
    return float(row["predicted_price"]), "publie"


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100.0)


def main() -> int:
    print("=" * 80)
    print("EVALUATION PREDICTIONS -> PRIX REELS")
    print("=" * 80)

    settings = get_settings()
    sb = create_client(settings.supabase_url, settings.supabase_key)
    monitor = PerformanceMonitor(supabase_client=sb)

    since = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    rows = []
    offset = 0
    page_size = 1000
    while True:
        page = (
            sb.table("predictions")
            .select("*")
            .gte("created_at", since)
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = page.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    print(f"Predictions candidates: {len(rows)}")

    today = datetime.now(timezone.utc).date().isoformat()
    prices_by_market: dict[str, dict[str, float]] = {}

    def prices_for(market_id: str, table_name: str) -> dict[str, float]:
        if market_id in prices_by_market:
            return prices_by_market[market_id]
        response = (
            sb.table(table_name)
            .select("date,price")
            .gte("date", since[:10])
            .order("date", desc=False)
            .limit(5000)
            .execute()
        )
        by_day = {
            str(row["date"])[:10]: float(row["price"])
            for row in (response.data or [])
        }
        prices_by_market[market_id] = by_day
        return by_day

    def training_day(version: str) -> str | None:
        match = re.search(r"(20\d{6})", version or "")
        if not match:
            return None
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    groups: dict[tuple, dict] = defaultdict(
        lambda: {
            "y_true": [],
            "y_pred": [],
            "y_lo": [],
            "y_hi": [],
            "origin": [],
            "failed": [],
            "targets": [],
            "source": [],
        }
    )
    seen = set()
    skipped = 0
    for row in rows:
        target = row.get("target_date")
        origin = row.get("origin_price")
        if not target or origin is None:
            skipped += 1
            continue
        target_day = str(target)[:10]
        if target_day > today:
            continue
        version = str(row.get("model_version") or "")
        trained = training_day(version)
        if trained is None or target_day <= trained:
            skipped += 1
            continue
        market_cfg = resolve_api_market(str(row.get("market") or ""))
        if market_cfg is None:
            skipped += 1
            continue
        origin_day = str(row.get("origin_date") or "")[:10]
        identity = (market_cfg.market_id, origin_day, int(row["horizon"]), version)
        if identity in seen:
            continue
        seen.add(identity)
        actual = prices_for(market_cfg.market_id, market_cfg.price_table).get(target_day)
        if actual is None:
            continue
        key = (market_cfg.market_id, int(row["horizon"]), version)
        bucket = groups[key]
        forecast, source = scored_forecast(row)
        bucket["y_true"].append(actual)
        bucket["y_pred"].append(forecast)
        bucket["source"].append(source)
        bucket["y_lo"].append(float(row["lower_bound"]))
        bucket["y_hi"].append(float(row["upper_bound"]))
        bucket["origin"].append(float(origin))
        bucket["failed"].append(bool(row.get("feature_failure")))
        bucket["targets"].append(target_day)

    print(f"Lignes ignorees (sans date cible, prix d'origine ou marche): {skipped}")
    if not groups:
        print("[AVERTISSEMENT] Pas assez de paires pour calculer des metriques")
        return 0

    stored = 0
    for (market_id, horizon, version), bucket in sorted(groups.items()):
        n = len(bucket["y_true"])
        from src.validation.acceptance import effective_sample_size, load_acceptance

        rules = load_acceptance()
        n_eff = effective_sample_size(n, horizon, int(rules.get("step_size", 5)))
        fallback_rate = float(sum(bucket["failed"]) / n) if n else 0.0
        period = (
            f"{min(bucket['targets'])} -> {max(bucket['targets'])}"
            if bucket["targets"]
            else "—"
        )
        model_mape = _mape(bucket["y_true"], bucket["y_pred"])
        naive_mape = _mape(bucket["y_true"], bucket["origin"])
        gain = (naive_mape - model_mape) / naive_mape if naive_mape > 0 else None
        gain_txt = "n/a" if gain is None else f"{gain:.1%}"
        followed = "candidat" if "candidat" in bucket["source"] else "prix publie"
        print(
            f"Paires {market_id} h{horizon} {version}: n={n} n_eff={n_eff:.1f} "
            f"repli={fallback_rate:.1%} periode={period} mesure=prospective suivi={followed}"
        )
        print(
            f"  MAPE suivi {model_mape:.2f}%  MAPE cloture {naive_mape:.2f}%  gain={gain_txt}"
        )
        if followed == "prix publie":
            print("  Pas de candidate_price : le suivi est le prix publie, souvent la cloture.")
        if n < 3:
            continue
        metrics = monitor.compute_metrics(
            np.array(bucket["y_true"], dtype=float),
            np.array(bucket["y_pred"], dtype=float),
            np.array(bucket["y_lo"], dtype=float),
            np.array(bucket["y_hi"], dtype=float),
            origin_price=np.array(bucket["origin"], dtype=float),
        )
        insert = {
            "model_version": version,
            "market": market_id,
            "horizon": horizon,
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "mape": float(metrics["mape"]),
            "directional_accuracy": float(metrics["directional_accuracy"]),
            "coverage_rate": float(metrics["coverage_rate"]),
            "mean_interval_width": float(metrics.get("mean_interval_width") or 0),
        }
        sb.table("model_metrics").insert(insert).execute()
        stored += 1
        print(f"  MAPE {insert['mape']:.4f}  direction {insert['directional_accuracy']:.3f}")

    print(f"\nGroupes enregistres: {stored}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
