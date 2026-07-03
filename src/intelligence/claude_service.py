"""
Client Anthropic avec routage Sonnet (standard) / Opus (avancé).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from loguru import logger

from config.settings import get_config, get_settings

SYSTEM_PROMPT = """Tu es un analyste financier specialise en matieres premieres (cacao, cafe robusta).
Tu rediges des notes de marche pour des decideurs et des traders — pas pour des data scientists.

STYLE OBLIGATOIRE — langage financier professionnel :
- Ecris comme une note de recherche ou un flash marche (Bloomberg / Reuters style).
- Parle de cours, tendance, pression vendeuse/acheteuse, niveaux cles, correction, consolidation, rebond, exposition, stop.
- Utilise des formulations naturelles : "a un jour", "a une semaine", "a un mois" (jamais "J+1", "J+7").
- Pour l'incertitude, dis "fourchette de prix large", "visibilite limitee", "scenario encore ouvert" — jamais "intervalle de confiance", "IC 90%", "GARCH", "modele hybride", "Prophet", "XGBoost", "N-HiTS", "FinBERT", "conformal", "walk-forward".
- Pour les alertes techniques, reformule en langage de marche : "le cours a perce le seuil des X $", "pression vendeuse sous les X $". Ne cite pas TradingView, Pine Script, ni "alerte technique".
- N'explique jamais comment le prix a ete calcule. Interprete uniquement la lecture de marche.

Tu ne recalcules JAMAIS les prix : tu t'appuies uniquement sur les niveaux fournis.

Reponds UNIQUEMENT en JSON valide (sans markdown) avec cette structure:
{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": "low" | "medium" | "high",
  "trend": "bullish" | "bearish" | "neutral",
  "summary": "2-3 phrases en francais, style financier",
  "outlook_7d": "1-2 phrases sur la semaine a venir",
  "key_levels": {"support": number|null, "resistance": number|null},
  "risks": ["risque marche 1", "risque marche 2"],
  "recommendation": "conseil actionnable en 1 phrase (exposition, prudence, niveaux a surveiller)",
  "disclaimer": "Analyse informative uniquement. Ne constitue pas un conseil financier."
}
Reste concis, factuel, en francais."""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


class ClaudeService:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.anthropic_api_key
        self.model_standard = get_config("claude.model_standard", "claude-sonnet-4-6")
        self.model_advanced = get_config("claude.model_advanced", "claude-opus-4-8")
        self.max_tokens_standard = int(get_config("claude.max_tokens_standard", 1200))
        self.max_tokens_advanced = int(get_config("claude.max_tokens_advanced", 2500))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate_brief(
        self,
        user_prompt: str,
        *,
        advanced: bool = False,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY non configuree")

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        model = self.model_advanced if advanced else self.model_standard
        max_tokens = self.max_tokens_advanced if advanced else self.max_tokens_standard

        logger.info(f"Claude brief: model={model}, advanced={advanced}")
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        parsed = _extract_json(text)
        parsed["_meta"] = {
            "model": model,
            "mode": "advanced" if advanced else "standard",
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        return parsed
