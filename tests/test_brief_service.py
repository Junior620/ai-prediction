"""Tests unitaires BriefService (quota Opus, cache)."""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.intelligence.brief_service import BriefService


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def incr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) + 1)
        return int(self.store[key])

    def expire(self, key, ttl):
        pass

    def pipeline(self):
        return _Pipeline(self)


class _Pipeline:
    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._ops = []

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))
        return self

    def execute(self):
        for op, key, *rest in self._ops:
            if op == "incr":
                self._redis.incr(key)
            elif op == "expire":
                self._redis.expire(key, rest[0])


class FakeRedisCache:
    def __init__(self):
        self.redis_client = FakeRedis()


@pytest.fixture
def brief_service():
    with patch("src.intelligence.brief_service.get_config") as mock_cfg:
        mock_cfg.side_effect = lambda k, d=None: {
            "claude.opus_daily_limit": 3,
            "claude.brief_cache_ttl": 86400,
        }.get(k, d)
        svc = BriefService(redis_cache=FakeRedisCache())
        svc.claude = MagicMock()
        svc.claude.available = True
        svc.claude.generate_brief.return_value = {
            "signal": "HOLD",
            "confidence": "medium",
            "trend": "neutral",
            "summary": "Test",
            "risks": [],
            "recommendation": "Attendre",
            "disclaimer": "Info only",
        }
        yield svc


def test_opus_quota_decrements(brief_service):
    assert brief_service.get_opus_remaining("user1") == 3
    brief_service._consume_opus_quota("user1")
    assert brief_service.get_opus_remaining("user1") == 2


def test_brief_cache_roundtrip(brief_service):
    payload = {"brief": {"signal": "BUY"}, "generated_at": datetime.utcnow().isoformat()}
    brief_service._set_cached_brief("ICE_NY", "standard", payload)
    cached = brief_service._get_cached_brief("ICE_NY", "standard")
    assert cached is not None
    assert cached["brief"]["signal"] == "BUY"


def test_opus_quota_blocks_when_exhausted(brief_service):
    key = brief_service._opus_key("user2")
    brief_service.redis.redis_client.store[key] = "3"
    assert brief_service.get_opus_remaining("user2") == 0


def test_build_prompt_injects_alert_context():
    """L'alerte TradingView doit apparaitre dans le prompt envoye a Claude."""
    from src.intelligence.brief_service import _build_prompt
    from src.models.market_registry import MarketConfig

    cfg = MarketConfig(
        market_id="cocoa",
        display_name="Cacao (ICE NY)",
        price_table="cocoa_prices",
        source="yahoo_finance",
        unit="USD/MT",
        price_bounds=(1000.0, 15000.0),
        models_dir="models/cocoa",
        ensemble_weights_file="config/ensemble_weights.json",
        conformal_intervals_file="config/conformal_intervals.json",
        nhits_unique_id="cocoa_ice_ny",
        api_markets=["ICE_NY"],
        garch_enabled=False,
        yahoo_symbol="CC=F",
        contract_symbol="CC=F",
        tradingview_symbol="PEPPERSTONE:COCOA",
        tradingview_embed_symbol="PEPPERSTONE:COCOA",
    )

    predictions = [
        {
            "horizon": 1,
            "price": 4000.0,
            "confidence_interval": [3500.0, 4500.0],
            "confidence_level": 0.9,
            "components": {},
        }
    ]

    prompt = _build_prompt(
        cfg,
        "ICE_NY",
        current_price=5027.0,
        current_date="2026-07-02",
        predictions=predictions,
        sentiment_score=None,
        news_headlines=[],
        advanced=False,
        alert_context={
            "signal_type": "support_break",
            "price": 4995.5,
            "tf": "1D",
            "ticker": "ICEEUR:C1!",
            "indicator": "PineSupportBreak_v1",
            "message": "Cours sous le seuil des 5000",
            "timestamp": "2026-07-03T14:00:00Z",
            "trend": "bearish",
            "momentum": "strong_sell",
            "change_pct": -1.8,
            "rsi": 28.5,
            "price_vs_ma": "below",
            "support": 4980.0,
            "resistance": 5120.0,
            "volume_ratio": 1.7,
        },
    )

    assert "LECTURE DE MARCHE RECENTE" in prompt
    assert "contrat cacao Londres" in prompt
    assert "cassure d'un support" in prompt
    assert "$4,995.50" in prompt
    assert "Tendance de fond: baissiere" in prompt
    assert "Momentum court terme: tres vendeur" in prompt
    assert "zone de survente" in prompt
    assert "pression vendeuse" in prompt
    assert "Volumes: nettement superieurs" in prompt
    assert "a un jour" in prompt
    assert "J+1" not in prompt
    # Traduction financiere : pas de jargon indicateur dans le prompt Claude
    assert "RSI" not in prompt
    assert "rsi" not in prompt


def test_build_prompt_without_alert_context_has_no_alert_block():
    from src.intelligence.brief_service import _build_prompt
    from src.models.market_registry import MarketConfig

    cfg = MarketConfig(
        market_id="cocoa",
        display_name="Cacao (ICE NY)",
        price_table="cocoa_prices",
        source="yahoo_finance",
        unit="USD/MT",
        price_bounds=(1000.0, 15000.0),
        models_dir="models/cocoa",
        ensemble_weights_file="config/ensemble_weights.json",
        conformal_intervals_file="config/conformal_intervals.json",
        nhits_unique_id="cocoa_ice_ny",
        api_markets=["ICE_NY"],
        garch_enabled=False,
        yahoo_symbol="CC=F",
        contract_symbol="CC=F",
    )
    prompt = _build_prompt(
        cfg,
        "ICE_NY",
        current_price=5027.0,
        current_date="2026-07-02",
        predictions=[
            {
                "horizon": 1,
                "price": 4000.0,
                "confidence_interval": [3500.0, 4500.0],
                "confidence_level": 0.9,
                "components": {},
            }
        ],
        sentiment_score=None,
        news_headlines=[],
        advanced=False,
        alert_context=None,
    )
    assert "LECTURE DE MARCHE RECENTE" not in prompt
