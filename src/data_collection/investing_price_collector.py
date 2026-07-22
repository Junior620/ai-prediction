"""
Collecteur de prix Investing.com (HTTP via curl_cffi).

Investing.com protege les pages avec Cloudflare : Selenium headless est bloque
("verification de securite"). On utilise curl_cffi (empreinte Chrome) pour
recuperer le HTML, puis on parse le prix et le symbole du contrat actif
(ex. RCU6) pour suivre les rollovers.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Titre / H1 type : "London Coffee (RCU6)"
SYMBOL_PATTERN = re.compile(r"\(([A-Z]{2,4}[A-Z0-9]{1,4})\)")

PRICE_PATTERNS = [
    # DOM rendu SSR
    re.compile(
        r'data-test=["\']instrument-price-last["\'][^>]*>\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)',
        re.IGNORECASE,
    ),
    # JSON embarque dans la page
    re.compile(r'"last"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
    re.compile(r'"last_numeric"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
]


def _parse_price(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _extract_price(html: str) -> Optional[float]:
    for pattern in PRICE_PATTERNS:
        match = pattern.search(html)
        if not match:
            continue
        price = _parse_price(match.group(1))
        if price is not None and price > 0:
            return price
    return None


def _extract_symbol(html: str, fallback_symbol: str = "") -> str:
    # Prefer explicit London Coffee / instrument title form
    titled = re.search(
        r"(?:London Coffee|instrument-name)[^<]{0,80}\(([A-Z]{2,4}[A-Z0-9]{1,4})\)",
        html,
        re.IGNORECASE,
    )
    if titled:
        return titled.group(1)

    title_tag = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if title_tag:
        match = SYMBOL_PATTERN.search(title_tag.group(1))
        if match:
            return match.group(1)

    match = SYMBOL_PATTERN.search(html[:5000])
    if match:
        return match.group(1)

    return fallback_symbol


def fetch_investing_price(
    url: str,
    fallback_symbol: str = "",
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Scrape le dernier prix affiche sur une page instrument Investing.com.

    Returns:
        {"price": float, "symbol": str, "date": "YYYY-MM-DD", "source": "investing_com"}
        ou None en cas d'echec.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError as exc:
        logger.error(
            "curl_cffi manquant (%s). Installez: pip install curl_cffi",
            exc,
        )
        return None

    try:
        response = cffi_requests.get(
            url,
            impersonate="chrome131",
            timeout=timeout,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
    except Exception as exc:
        logger.error("Investing.com HTTP failed (%s): %s", url, exc)
        return None

    if response.status_code != 200:
        logger.error(
            "Investing.com HTTP %s sur %s (Cloudflare / bloque ?)",
            response.status_code,
            url,
        )
        return None

    html = response.text or ""
    if "Just a moment" in html or "cf-challenge" in html.lower():
        logger.error("Investing.com: challenge Cloudflare non resolu (%s)", url)
        return None

    price = _extract_price(html)
    if price is None:
        logger.error("Investing.com: prix introuvable sur %s", url)
        return None

    symbol = _extract_symbol(html, fallback_symbol=fallback_symbol)
    logger.info(
        "Investing.com OK: %s = %.2f (contrat %s)",
        url,
        price,
        symbol or "?",
    )

    return {
        "price": price,
        "symbol": symbol,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "investing_com",
    }
