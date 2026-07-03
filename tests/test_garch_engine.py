"""Tests du moteur de volatilité GARCH(1,1)-t."""

import numpy as np
import pandas as pd
import pytest

from src.models.garch_engine import fit_garch_forecast, std_t_ppf


@pytest.fixture(scope="module")
def synthetic_prices() -> pd.Series:
    """Série de prix avec clustering de volatilité (1000 points)."""
    rng = np.random.default_rng(42)
    n = 1000
    vol = np.abs(np.sin(np.arange(n) / 50)) * 0.02 + 0.01
    returns = rng.standard_normal(n) * vol
    return pd.Series(3500 * np.exp(np.cumsum(returns)))


class TestStdTPpf:
    def test_standardized_variance(self):
        # Le quantile standardisé doit être plus petit que le quantile t brut
        from scipy import stats
        nu = 5.7
        raw = stats.t.ppf(0.95, nu)
        std = std_t_ppf(0.95, nu)
        assert std < raw
        assert std == pytest.approx(raw * np.sqrt((nu - 2) / nu))

    def test_invalid_nu_raises(self):
        with pytest.raises(ValueError):
            std_t_ppf(0.95, 2.0)


class TestFitGarchForecast:
    def test_fit_returns_forecast(self, synthetic_prices):
        fc = fit_garch_forecast(synthetic_prices, horizons=(1, 7, 30))
        assert fc is not None
        assert fc.nu > 2
        assert fc.last_price == pytest.approx(float(synthetic_prices.iloc[-1]))

    def test_interval_widths_grow_with_horizon(self, synthetic_prices):
        fc = fit_garch_forecast(synthetic_prices, horizons=(1, 7, 30))
        w = {}
        for h in (1, 7, 30):
            lo, hi = fc.interval(h, 0.90)
            assert lo < fc.last_price < hi
            w[h] = hi - lo
        assert w[1] < w[7] < w[30]

    def test_interval_around_custom_price(self, synthetic_prices):
        fc = fit_garch_forecast(synthetic_prices, horizons=(1, 7, 30))
        lo, hi = fc.interval_around(3600.0, 7, 0.90)
        assert lo < 3600.0 < hi

    def test_higher_confidence_widens_interval(self, synthetic_prices):
        fc = fit_garch_forecast(synthetic_prices, horizons=(1, 7, 30))
        lo90, hi90 = fc.interval(7, 0.90)
        lo95, hi95 = fc.interval(7, 0.95)
        assert (hi95 - lo95) > (hi90 - lo90)

    def test_insufficient_data_returns_none(self):
        short = pd.Series(np.linspace(3000, 3100, 50))
        assert fit_garch_forecast(short) is None

    def test_annualized_volatility_positive(self, synthetic_prices):
        fc = fit_garch_forecast(synthetic_prices, horizons=(1, 7, 30))
        vol = fc.annualized_volatility(1)
        assert vol is not None and vol > 0
