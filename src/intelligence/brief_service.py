"""
Assemble le contexte marche et genere un brief Claude.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from supabase import Client

from config.settings import get_config
from src.intelligence.claude_service import ClaudeService
from src.models.data_models import NewsArticle
from src.models.market_registry import MarketConfig, resolve_api_market


def _build_prompt(
    market_cfg: MarketConfig,
    api_market: str,
    current_price: float,
    current_date: Optional[str],
    predictions: List[Dict[str, Any]],
    sentiment_score: Optional[float],
    news_headlines: List[str],
    advanced: bool,
    user_question: Optional[str] = None,
    alert_context: Optional[Dict[str, Any]] = None,
) -> str:
    horizon_labels = {1: "a un jour", 7: "a une semaine", 30: "a un mois"}
    preds_text = []
    for p in predictions:
        lo, hi = p["confidence_interval"]
        comps = p.get("components") or {}
        label = horizon_labels.get(p["horizon"], f"a {p['horizon']} jours")
        line = (
            f"  Horizon {label}: cours vise ${p['price']:,.2f} "
            f"(fourchette ${lo:,.0f} – ${hi:,.0f})"
        )
        if comps.get("garch_annualized_volatility") is not None:
            line += f" | volatilite annuelle estimee {comps['garch_annualized_volatility']:.1f}%"
        if comps.get("high_volatility_regime"):
            line += " | marche tres agite"
        preds_text.append(line)

    news_block = "\n".join(f"  - {h}" for h in news_headlines[:8]) or "  (aucune actualite recente)"

    extra = ""
    if advanced and user_question:
        extra = f"\nQuestion de l'utilisateur:\n{user_question}\n"

    alert_block = ""
    if alert_context:
        alert_block = _format_market_reading(alert_context)

    sentiment_label = "N/A"
    if sentiment_score is not None:
        if sentiment_score > 0.2:
            sentiment_label = "plutot positif"
        elif sentiment_score < -0.2:
            sentiment_label = "plutot negatif"
        else:
            sentiment_label = "neutre"

    return f"""Marche: {market_cfg.display_name}
Unite: {market_cfg.unit}
Cours actuel: ${current_price:,.2f} ({current_date or "aujourd'hui"})

Niveaux de prix anticipes (a interpreter, ne pas recalculer):
{chr(10).join(preds_text)}

Climat des actualites: {sentiment_label}

Titres recents:
{news_block}
{alert_block}{extra}
Redige le brief JSON en style financier professionnel.
Interdictions dans le texte final: jargon technique (modele, hybride, ML, IA, intervalle de confiance,
indicateurs, plateformes chartistes, noms d'algorithmes). Utilise uniquement "a un jour / a une semaine / a un mois".
Le signal doit rester coherent avec les niveaux anticipes et la largeur des fourchettes."""


def _format_market_reading(alert_context: Dict[str, Any]) -> str:
    """
    Traduit le payload TradingView en lecture de marche financiere.
    Claude recoit du langage de marche, pas du jargon d'indicateurs.
    """
    lines = ["", "=== LECTURE DE MARCHE RECENTE ==="]

    ticker = (alert_context.get("ticker") or "").upper()
    if "C1!" in ticker or ticker.endswith(":C1!"):
        lines.append("Source de lecture: contrat cacao Londres (ICE)")
    elif ticker:
        lines.append(f"Source de lecture: {alert_context.get('ticker')}")

    signal = (alert_context.get("signal_type") or "").lower()
    signal_map = {
        "buy": "pression acheteuse",
        "sell": "pression vendeuse",
        "support_break": "cassure d'un support",
        "resistance_break": "franchissement d'une resistance",
        "trend_change": "changement de tendance",
        "custom": "signal de marche",
    }
    if signal:
        lines.append(f"Evenement: {signal_map.get(signal, signal.replace('_', ' '))}")

    price = alert_context.get("price")
    if price not in (None, ""):
        lines.append(f"Cours au moment du signal: ${float(price):,.2f}")

    change_pct = alert_context.get("change_pct")
    if change_pct not in (None, ""):
        direction = "en hausse" if float(change_pct) >= 0 else "en baisse"
        lines.append(f"Variation de seance: {float(change_pct):+.2f}% ({direction})")

    trend = (alert_context.get("trend") or "").lower()
    trend_map = {
        "bullish": "haussiere",
        "bearish": "baissiere",
        "neutral": "neutre / sans direction claire",
    }
    if trend in trend_map:
        lines.append(f"Tendance de fond: {trend_map[trend]}")

    momentum = (alert_context.get("momentum") or "").lower()
    momentum_map = {
        "strong_buy": "tres acheteur",
        "buy": "plutot acheteur",
        "neutral": "equilibre",
        "sell": "plutot vendeur",
        "strong_sell": "tres vendeur",
    }
    if momentum in momentum_map:
        lines.append(f"Momentum court terme: {momentum_map[momentum]}")

    # RSI traduit en langage de marche (jamais exposer "RSI" a Claude dans le brief final)
    rsi = alert_context.get("rsi")
    if rsi not in (None, ""):
        rsi_f = float(rsi)
        if rsi_f >= 70:
            lines.append("Dynamique des prix: zone de surachat (risque de reprise vendeuse)")
        elif rsi_f <= 30:
            lines.append("Dynamique des prix: zone de survente (risque de rebond technique)")
        elif rsi_f >= 55:
            lines.append("Dynamique des prix: legerement orientee a la hausse")
        elif rsi_f <= 45:
            lines.append("Dynamique des prix: legerement orientee a la baisse")
        else:
            lines.append("Dynamique des prix: neutre")

    price_vs_ma = (alert_context.get("price_vs_ma") or "").lower()
    ma_map = {
        "above": "cours au-dessus des moyennes de reference (biais haussier)",
        "below": "cours sous les moyennes de reference (pression vendeuse)",
        "mixed": "cours entre les moyennes de reference (signal mixte)",
    }
    if price_vs_ma in ma_map:
        lines.append(f"Positionnement: {ma_map[price_vs_ma]}")

    support = alert_context.get("support")
    resistance = alert_context.get("resistance")
    if support not in (None, ""):
        lines.append(f"Support proche: ${float(support):,.2f}")
    if resistance not in (None, ""):
        lines.append(f"Resistance proche: ${float(resistance):,.2f}")

    volume_ratio = alert_context.get("volume_ratio")
    if volume_ratio not in (None, ""):
        vr = float(volume_ratio)
        if vr >= 1.5:
            lines.append("Volumes: nettement superieurs a la moyenne (mouvement confirme)")
        elif vr <= 0.7:
            lines.append("Volumes: en retrait par rapport a la moyenne (mouvement peu confirme)")
        else:
            lines.append("Volumes: dans la moyenne")

    message = alert_context.get("message")
    if message:
        lines.append(f"Contexte: {message}")

    lines.append(
        "Integre cette lecture de marche dans ton analyse. "
        "Ne cite aucun indicateur technique, plateforme ou outil. "
        "Reformule uniquement en langage financier (tendance, pression, niveaux, volumes)."
    )
    return "\n".join(lines) + "\n"


def _fetch_recent_news(supabase: Client, days: int = 7) -> List[str]:
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        resp = (
            supabase.table("news_articles")
            .select("title")
            .gte("published_at", cutoff.isoformat())
            .order("published_at", desc=True)
            .limit(12)
            .execute()
        )
        return [r["title"] for r in (resp.data or []) if r.get("title")]
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return []


class BriefService:
    def __init__(self, redis_cache=None):
        self.claude = ClaudeService()
        self.redis = redis_cache
        self.opus_limit = int(get_config("claude.opus_daily_limit", 3))
        self.brief_ttl = int(get_config("claude.brief_cache_ttl", 86400))

    def _cache_key(self, market: str, mode: str) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"brief:{market}:{mode}:{today}"

    def _opus_key(self, user_id: str) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"claude:opus:{user_id}:{today}"

    def get_opus_remaining(self, user_id: str) -> int:
        if not self.redis or not self.redis.redis_client:
            return self.opus_limit
        try:
            used = int(self.redis.redis_client.get(self._opus_key(user_id)) or 0)
            return max(0, self.opus_limit - used)
        except Exception:
            return self.opus_limit

    def _consume_opus_quota(self, user_id: str) -> None:
        if not self.redis or not self.redis.redis_client:
            return
        key = self._opus_key(user_id)
        pipe = self.redis.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 86400)
        pipe.execute()

    def _get_cached_brief(self, market: str, mode: str) -> Optional[Dict[str, Any]]:
        if not self.redis or not self.redis.redis_client:
            return None
        try:
            raw = self.redis.redis_client.get(self._cache_key(market, mode))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _set_cached_brief(self, market: str, mode: str, payload: Dict[str, Any]) -> None:
        if not self.redis or not self.redis.redis_client:
            return
        try:
            self.redis.redis_client.setex(
                self._cache_key(market, mode),
                self.brief_ttl,
                json.dumps(payload, default=str),
            )
        except Exception as e:
            logger.warning(f"Brief cache write failed: {e}")

    def generate(
        self,
        *,
        api_market: str,
        predictor,
        supabase: Client,
        user_id: str = "system",
        advanced: bool = False,
        user_question: Optional[str] = None,
        force_refresh: bool = False,
        alert_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.claude.available:
            raise RuntimeError(
                "Service Claude indisponible: definissez ANTHROPIC_API_KEY dans .env"
            )

        market_cfg = resolve_api_market(api_market)
        if market_cfg is None:
            raise ValueError(f"Marche inconnu: {api_market}")

        mode = "advanced" if advanced else "standard"

        if not advanced and not force_refresh:
            cached = self._get_cached_brief(api_market, mode)
            if cached:
                cached["cached"] = True
                return cached

        if advanced:
            remaining = self.get_opus_remaining(user_id)
            if remaining <= 0:
                raise PermissionError(
                    f"Quota Opus epuise ({self.opus_limit}/jour). Reessayez demain."
                )

        if predictor is None:
            raise RuntimeError(f"Predictor non charge pour {market_cfg.market_id}")

        include_sentiment = market_cfg.market_id == "cocoa"
        recent_news_objs: List[NewsArticle] = []
        news_headlines: List[str] = []

        if include_sentiment and supabase:
            try:
                cutoff = datetime.utcnow() - timedelta(days=7)
                resp = (
                    supabase.table("news_articles")
                    .select("*")
                    .gte("published_at", cutoff.isoformat())
                    .order("published_at", desc=True)
                    .limit(20)
                    .execute()
                )
                for row in resp.data or []:
                    news_headlines.append(row.get("title", ""))
                    recent_news_objs.append(
                        NewsArticle(
                            id=row["id"],
                            source=row["source"],
                            title=row["title"],
                            content=row.get("content", ""),
                            published_at=datetime.fromisoformat(
                                row["published_at"].replace("Z", "+00:00")
                            ),
                            url=row.get("url", ""),
                            keywords=row.get("keywords", []),
                            sentiment_score=row.get("sentiment_score"),
                            is_high_risk=row.get("is_high_risk"),
                        )
                    )
            except Exception as e:
                logger.warning(f"News load failed: {e}")

        preds = predictor.predict(
            horizons=[1, 7, 30],
            recent_news=recent_news_objs if include_sentiment else [],
        )

        sentiment_score = None
        if include_sentiment and recent_news_objs:
            try:
                sentiment_score = predictor.nlp_analyzer.aggregate_sentiment(recent_news_objs)
            except Exception:
                pass

        # Prix actuel depuis Supabase
        current_price = preds[0].price
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            hist = (
                supabase.table(market_cfg.price_table)
                .select("date, price")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if hist.data:
                current_price = hist.data[0]["price"]
                current_date = hist.data[0]["date"]
        except Exception:
            pass

        pred_dicts = [
            {
                "horizon": p.horizon,
                "price": p.price,
                "confidence_interval": list(p.confidence_interval),
                "confidence_level": p.confidence_level,
                "components": p.components or {},
            }
            for p in preds
        ]

        prompt = _build_prompt(
            market_cfg,
            api_market,
            current_price,
            current_date,
            pred_dicts,
            sentiment_score,
            news_headlines,
            advanced,
            user_question,
            alert_context=alert_context,
        )

        brief = self.claude.generate_brief(prompt, advanced=advanced)

        if advanced:
            self._consume_opus_quota(user_id)

        result = {
            "market": api_market,
            "market_display_name": market_cfg.display_name,
            "unit": market_cfg.unit,
            "tradingview_symbol": market_cfg.tradingview_symbol,
            "current_price": current_price,
            "current_date": current_date,
            "model_version": predictor.model_version,
            "sentiment_score": sentiment_score,
            "predictions": pred_dicts,
            "brief": brief,
            "mode": mode,
            "opus_remaining": self.get_opus_remaining(user_id),
            "generated_at": datetime.utcnow().isoformat(),
            "cached": False,
        }

        if not advanced:
            self._set_cached_brief(api_market, mode, result)

        return result
