"""
Market registry: per-market configuration (data table, models dir, calibration files).

Markets are declared in config/config.yaml under the `markets` section.
API market identifiers (e.g. "ICE_NY", "COFFEE_ROBUSTA") map to internal
market ids (e.g. "cocoa", "coffee_robusta") via each market's `api_markets` list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import get_config

DEFAULT_MARKET_ID = "cocoa"


@dataclass
class MarketConfig:
    market_id: str
    display_name: str
    price_table: str
    source: str
    unit: str
    price_bounds: Tuple[float, float]
    models_dir: str
    ensemble_weights_file: str
    conformal_intervals_file: str
    nhits_unique_id: str
    api_markets: List[str] = field(default_factory=list)
    garch_enabled: bool = False
    yahoo_symbol: Optional[str] = None
    investing_url: Optional[str] = None
    contract_symbol: Optional[str] = None
    tradingview_symbol: Optional[str] = None
    tradingview_embed_symbol: Optional[str] = None
    tradingview_embed_label: Optional[str] = None
    tradingview_alert_symbol: Optional[str] = None


def _build_market_config(market_id: str, raw: Dict) -> MarketConfig:
    bounds = raw.get("price_bounds", {}) or {}
    return MarketConfig(
        market_id=market_id,
        display_name=raw.get("display_name", market_id),
        price_table=raw.get("price_table", "cocoa_prices"),
        source=raw.get("source", "yahoo_finance"),
        unit=raw.get("unit", "USD/MT"),
        price_bounds=(float(bounds.get("min", 1000)), float(bounds.get("max", 10000))),
        models_dir=raw.get("models_dir", "models"),
        ensemble_weights_file=raw.get("ensemble_weights_file", "config/ensemble_weights.json"),
        conformal_intervals_file=raw.get(
            "conformal_intervals_file", "config/conformal_intervals.json"
        ),
        nhits_unique_id=raw.get("nhits_unique_id", f"{market_id}_series"),
        api_markets=list(raw.get("api_markets", []) or []),
        garch_enabled=bool(raw.get("garch_enabled", False)),
        yahoo_symbol=raw.get("yahoo_symbol"),
        investing_url=raw.get("investing_url"),
        contract_symbol=raw.get("contract_symbol"),
        tradingview_symbol=raw.get("tradingview_symbol"),
        tradingview_embed_symbol=raw.get("tradingview_embed_symbol"),
        tradingview_embed_label=raw.get("tradingview_embed_label"),
        tradingview_alert_symbol=raw.get("tradingview_alert_symbol"),
    )


def load_all_markets() -> Dict[str, MarketConfig]:
    """Return all markets declared in config.yaml, keyed by market id."""
    raw_markets = get_config("markets", {}) or {}
    return {
        market_id: _build_market_config(market_id, raw or {})
        for market_id, raw in raw_markets.items()
    }


def get_market_config(market_id: str = DEFAULT_MARKET_ID) -> MarketConfig:
    """Return config for a market id (e.g. "cocoa", "coffee_robusta")."""
    markets = load_all_markets()
    if market_id not in markets:
        raise KeyError(
            f"Unknown market '{market_id}'. Available: {sorted(markets)}"
        )
    return markets[market_id]


def resolve_api_market(api_market: str) -> Optional[MarketConfig]:
    """
    Map an API market identifier (e.g. "ICE_NY", "COFFEE_ROBUSTA") to its
    MarketConfig. Returns None if no market declares this identifier.
    """
    normalized = (api_market or "").strip().upper()
    for cfg in load_all_markets().values():
        if normalized in {m.upper() for m in cfg.api_markets}:
            return cfg
    return None


def list_api_markets() -> List[str]:
    """All valid API market identifiers across declared markets."""
    identifiers: List[str] = []
    for cfg in load_all_markets().values():
        identifiers.extend(cfg.api_markets)
    return identifiers
