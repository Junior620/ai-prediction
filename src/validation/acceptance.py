"""Predeclared rules for accepting one forecast horizon.

Thresholds live in config/acceptance.json and are not tuned after a run.
Overlapping origins are dependent, so uncertainty uses a moving-block
bootstrap rather than a binomial bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_ACCEPTANCE = Path("config/acceptance.json")


def load_acceptance(path: Optional[str] = None) -> Dict[str, Any]:
    file_path = Path(path) if path else DEFAULT_ACCEPTANCE
    if not file_path.exists():
        raise FileNotFoundError(f"Acceptance file missing: {file_path}")
    with open(file_path, encoding="utf-8") as handle:
        return json.load(handle)


def temporal_overlap(horizon: int, step: int) -> int:
    """Neighbouring origins still covered by one forecast. None at one session."""
    if int(horizon) <= 1:
        return 0
    return int(math.ceil(int(horizon) / max(int(step), 1)))


def effective_sample_size(n: int, horizon: int, step: int) -> float:
    """Observations divided by one plus the temporal overlap."""
    if n <= 0:
        return 0.0
    return float(n) / (1.0 + temporal_overlap(horizon, step))


def block_length(horizon: int, step: int) -> int:
    """Block at least as long as the horizon divided by the step."""
    return max(1, int(math.ceil(int(horizon) / max(int(step), 1))))


def moving_block_ci(
    values: Sequence[float],
    length: int,
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the mean, resampling contiguous blocks."""
    sample = np.asarray(list(values), dtype=float)
    sample = sample[np.isfinite(sample)]
    n = int(len(sample))
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(sample[0]), float(sample[0])
    block = max(1, min(int(length), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    means = np.empty(replicates, dtype=float)
    for i in range(replicates):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        drawn = np.concatenate([sample[s : s + block] for s in starts])[:n]
        means[i] = float(np.mean(drawn))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def failure_mask(frame: pd.DataFrame) -> np.ndarray:
    """A row is a fallback when the model did not produce a usable price.

    The mask is the union of a recorded feature failure, a non-finite
    published price, and a required engine that was absent.
    """
    n = int(len(frame))
    failed = np.zeros(n, dtype=bool)
    if n == 0:
        return failed
    if "feature_failure" in frame.columns:
        failed |= frame["feature_failure"].fillna(False).astype(bool).to_numpy()
    if "engine_missing" in frame.columns:
        failed |= frame["engine_missing"].fillna(False).astype(bool).to_numpy()
    if "published_pred" in frame.columns:
        raw = pd.to_numeric(frame["published_pred"], errors="coerce").to_numpy(dtype=float)
        failed |= ~np.isfinite(raw)
    else:
        failed |= True
    return failed


def _published_column(frame: pd.DataFrame) -> pd.Series:
    """Price that would have been shown. A missing price is the close."""
    if "published_pred" not in frame.columns:
        published = pd.Series(np.nan, index=frame.index, dtype=float)
    else:
        published = pd.to_numeric(frame["published_pred"], errors="coerce")
    origin = pd.to_numeric(frame["origin_price"], errors="coerce")
    failed = failure_mask(frame)
    published = published.mask(failed, origin)
    return published.where(published.notna(), origin)


def _served_band(frame: pd.DataFrame, lower_name: str, upper_name: str):
    """Final bounds already stored on the rows. Missing columns are not invented."""
    if lower_name not in frame.columns or upper_name not in frame.columns:
        return None, None
    lower = pd.to_numeric(frame[lower_name], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame[upper_name], errors="coerce").to_numpy(dtype=float)
    return lower, upper


def _margin_width(entry: Optional[Dict[str, Any]]) -> float:
    """Width of a margin frozen before the test. Missing margins are not refit."""
    if not entry:
        return float("nan")
    lower = entry.get("margin_lower")
    upper = entry.get("margin_upper")
    if lower is None or upper is None:
        return float("nan")
    lower_f = float(lower)
    upper_f = float(upper)
    if not np.isfinite(lower_f) or not np.isfinite(upper_f):
        return float("nan")
    return lower_f + upper_f


def evaluate_horizon(
    frame: pd.DataFrame,
    horizon: int,
    acceptance: Dict[str, Any],
    step: Optional[int] = None,
    frozen_margin: Optional[Dict[str, Any]] = None,
    frozen_naive_margin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide one horizon. Fallbacks stay in the primary score as the close.

    Coverage and band width use margins frozen on the calibration slice.
    They are never refit on the rows being judged.
    """
    step_size = int(step if step is not None else acceptance.get("step_size", 5))
    subset = frame[frame["horizon"] == int(horizon)].copy()
    subset = subset.dropna(subset=["actual", "origin_price", "naive_pred"])
    subset = subset.sort_values("origin_date")
    n = int(len(subset))
    overlap = temporal_overlap(horizon, step_size)
    n_eff = effective_sample_size(n, horizon, step_size)
    empty = {
        "horizon": int(horizon),
        "validated": False,
        "reason": "Aucune observation commune.",
        "n": 0,
        "n_effective": 0.0,
        "overlap": overlap,
        "fallback_rate": None,
        "model_mape": None,
        "naive_mape": None,
        "relative_gain": None,
        "gap_ci": [None, None],
        "measurement": "procedure",
    }
    if n == 0:
        return empty

    published = _published_column(subset).to_numpy(dtype=float)
    actual = subset["actual"].to_numpy(dtype=float)
    naive = subset["naive_pred"].to_numpy(dtype=float)
    failed = failure_mask(subset)
    fallback_rate = float(np.mean(failed)) if n else 0.0

    model_ape = np.abs((actual - published) / actual) * 100.0
    naive_ape = np.abs((actual - naive) / actual) * 100.0
    model_mape = float(np.mean(model_ape))
    naive_mape = float(np.mean(naive_ape))
    relative_gain = None
    if naive_mape > 0:
        relative_gain = (naive_mape - model_mape) / naive_mape

    gap = model_ape - naive_ape
    length = block_length(horizon, step_size)
    replicates = int(acceptance.get("bootstrap_replicates", 1000))
    alpha = float(acceptance.get("bootstrap_alpha", 0.05))
    gap_lo, gap_hi = moving_block_ci(gap, length, replicates, alpha, seed=int(horizon))

    coverage_level = float(acceptance.get("coverage_level", 0.90))
    del frozen_margin, frozen_naive_margin
    model_lower, model_upper = _served_band(subset, "published_lower", "published_upper")
    naive_lower, naive_upper = _served_band(subset, "naive_lower", "naive_upper")
    model_bounds_ok = (
        model_lower is not None
        and bool(np.isfinite(model_lower).all())
        and bool(np.isfinite(model_upper).all())
    )
    naive_bounds_ok = (
        naive_lower is not None
        and bool(np.isfinite(naive_lower).all())
        and bool(np.isfinite(naive_upper).all())
    )
    if model_bounds_ok:
        covered = (actual >= model_lower) & (actual <= model_upper)
        model_width = float(np.mean(model_upper - model_lower))
    else:
        covered = np.zeros(n, dtype=bool)
        model_width = float("nan")
    naive_width = (
        float(np.mean(naive_upper - naive_lower)) if naive_bounds_ok else float("nan")
    )
    cov_lo, cov_hi = moving_block_ci(covered.astype(float), length, replicates, alpha, seed=1000 + int(horizon))

    spoken = ~failed
    secondary = None
    if spoken.any():
        secondary = float(np.mean(np.abs((actual[spoken] - published[spoken]) / actual[spoken]) * 100.0))

    min_gain = float(acceptance.get("min_relative_mape_gain", 0.10))
    max_fallback = float(acceptance.get("max_fallback_rate", 0.20))
    min_observations = int(acceptance.get("min_observations", 30))
    min_blocks = float(acceptance.get("min_temporal_blocks", 6))
    reasons = []
    if n < min_observations:
        reasons.append("Le nombre d'observations est sous le minimum écrit à l'avance.")
    if (n / float(length)) < min_blocks:
        reasons.append("Le nombre de blocs temporels est sous le minimum écrit à l'avance.")
    if not (relative_gain is not None and relative_gain >= min_gain):
        reasons.append("Le gain sur la clôture inchangée est inférieur au minimum utile.")
    if not (np.isfinite(gap_hi) and gap_hi < 0):
        reasons.append("L'intervalle par blocs sur l'écart n'exclut pas zéro.")
    if fallback_rate >= max_fallback:
        reasons.append("Le taux de repli vers la clôture dépasse le plafond.")
    if not model_bounds_ok:
        reasons.append("Bornes finales absentes.")
    elif not naive_bounds_ok:
        reasons.append("Bornes finales de la clôture inchangée absentes.")
    elif not (np.isfinite(model_width) and np.isfinite(naive_width) and model_width < naive_width):
        reasons.append("La bande n'est pas plus étroite que celle de la clôture inchangée.")
    elif not (np.isfinite(cov_hi) and cov_hi >= coverage_level):
        reasons.append("L'intervalle par blocs de la couverture hors échantillon reste sous la cible.")

    return {
        "horizon": int(horizon),
        "validated": len(reasons) == 0,
        "reason": " ".join(reasons) if reasons else "Les critères écrits à l'avance sont tenus.",
        "n": n,
        "n_effective": n_eff,
        "overlap": overlap,
        "fallback_rate": fallback_rate,
        "model_mape": model_mape,
        "naive_mape": naive_mape,
        "relative_gain": relative_gain,
        "gap_ci": [gap_lo, gap_hi],
        "coverage": float(np.mean(covered)),
        "coverage_ci": [cov_lo, cov_hi],
        "model_width": model_width,
        "naive_width": naive_width,
        "secondary_mape_when_model_spoke": secondary,
        "measurement": "procedure",
    }
