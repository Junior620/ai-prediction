"""
Moteur de volatilité GARCH(1,1)-t.

Adapté du dépôt F5F0A0/coffee-futures-forecasting-model : fit sur les
log-returns quotidiens, intervalles construits avec les quantiles de la
Student-t STANDARDISÉE (variance 1) pour matcher l'échelle des prévisions
de variance d'arch_model.

Rôle dans le système : couche volatilité/intervalles complémentaire aux
intervalles conformes (le point forecast GARCH zero-mean est plat, donc
inutile comme moteur de prix). L'intervalle final du predictor prend, par
borne, le plus large entre conforme et GARCH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Au-delà de ce seuil de volatilité annualisée, on considère un régime agité.
HIGH_VOL_ANNUALIZED_THRESHOLD = 40.0  # en %


def std_t_ppf(p: float, nu: float) -> float:
    """
    Quantile de la Student-t STANDARDISÉE (variance = 1).

    arch_model prévoit la variance en échelle standardisée ; utiliser
    scipy.stats.t.ppf brut (variance nu/(nu-2)) gonflerait les intervalles
    d'environ 20 % à nu ~ 5.7.
    """
    if nu <= 2:
        raise ValueError(f"Variance Student-t indéfinie pour nu={nu}")
    return float(stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu))


@dataclass
class GarchForecast:
    """Prévision de volatilité par horizon (jours ouvrés)."""

    cum_sd: Dict[int, float]          # écart-type cumulé des log-returns par horizon
    ann_vol: Dict[int, float]         # volatilité annualisée (%) par horizon
    nu: float                         # degrés de liberté Student-t
    last_price: float

    def interval(self, horizon: int, confidence_level: float = 0.90) -> Optional[Tuple[float, float]]:
        """Intervalle de prédiction (lower, upper) autour du dernier prix."""
        sd = self.cum_sd.get(horizon)
        if sd is None or not np.isfinite(sd):
            return None
        q = std_t_ppf(0.5 + confidence_level / 2.0, self.nu)
        lower = self.last_price * float(np.exp(-q * sd))
        upper = self.last_price * float(np.exp(+q * sd))
        return lower, upper

    def interval_around(
        self, price: float, horizon: int, confidence_level: float = 0.90
    ) -> Optional[Tuple[float, float]]:
        """Même intervalle multiplicatif, centré sur un prix prédit arbitraire."""
        sd = self.cum_sd.get(horizon)
        if sd is None or not np.isfinite(sd):
            return None
        q = std_t_ppf(0.5 + confidence_level / 2.0, self.nu)
        return price * float(np.exp(-q * sd)), price * float(np.exp(+q * sd))

    def annualized_volatility(self, horizon: int = 1) -> Optional[float]:
        return self.ann_vol.get(horizon)

    def high_volatility_regime(self, horizon: int = 1) -> bool:
        vol = self.ann_vol.get(horizon)
        return vol is not None and vol >= HIGH_VOL_ANNUALIZED_THRESHOLD


def fit_garch_forecast(
    prices: pd.Series,
    horizons: Tuple[int, ...] = (1, 7, 30),
    min_observations: int = 250,
) -> Optional[GarchForecast]:
    """
    Fit GARCH(1,1)-t (zero mean) sur les log-returns et prévoit la volatilité.

    Args:
        prices: série de prix quotidiens, ordonnée par date croissante.
        horizons: horizons (jours ouvrés) pour lesquels produire cum_sd.
        min_observations: minimum de points requis pour un fit fiable.

    Returns:
        GarchForecast, ou None si les données sont insuffisantes ou le fit échoue.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("Package 'arch' non installé — GARCH désactivé")
        return None

    values = pd.Series(prices).astype(float).dropna().values
    if len(values) < min_observations:
        logger.warning(
            "GARCH: %d observations < %d requis — skip", len(values), min_observations
        )
        return None

    # Returns en pourcentage pour la stabilité numérique de l'optimiseur.
    r_pct = np.diff(np.log(values)) * 100.0
    r_pct = r_pct[np.isfinite(r_pct)]
    if len(r_pct) < min_observations - 1:
        return None

    max_h = max(horizons)
    try:
        am = arch_model(r_pct, mean="Zero", vol="Garch", p=1, q=1, dist="t")
        fit = am.fit(disp="off", show_warning=False)
        var_fc_pct2 = fit.forecast(horizon=max_h).variance.iloc[-1].values
    except Exception as exc:
        logger.error("GARCH fit failed: %s", exc)
        return None

    step_var = var_fc_pct2 / 10_000.0  # percent^2 -> decimal^2
    cum_var = np.cumsum(step_var)
    nu = float(fit.params.get("nu", 8.0))

    cum_sd = {h: float(np.sqrt(cum_var[h - 1])) for h in horizons if h <= max_h}
    ann_vol = {
        h: float(np.sqrt(step_var[h - 1] * 252.0) * 100.0) for h in horizons if h <= max_h
    }

    return GarchForecast(
        cum_sd=cum_sd,
        ann_vol=ann_vol,
        nu=nu,
        last_price=float(values[-1]),
    )
