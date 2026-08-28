"""
Collecteur prix cafe ONCC Cameroun (Arabica + Robusta, FCFA/kg FOB).

Fallbacks : selecteurs CSS du guide utilisateur, puis recherche par libelle.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ONCC_URL = "https://www.oncc.cm/prices"

ARABICA_CSS = (
    "body > main > div > div:nth-child(5) > div > div > "
    "section:nth-child(2) > section > section:nth-child(2) > span:first-child"
)
ROBUSTA_CSS = (
    "body > main > div > div:nth-child(6) > div > div > "
    "section:nth-child(2) > section > section:nth-child(2) > span:first-child"
)

PRICE_INT_RE = re.compile(r"[\d\s,]+")


@dataclass
class OnccPrice:
    product: str  # arabica | robusta
    price: float
    date: str
    unit: str = "FCFA/KG FOB ONCC"
    source: str = "oncc"
    trend: str = "stable"
    change_pct: float = 0.0
    strategy: str = ""


@dataclass
class OnccCollectionResult:
    prices: List[OnccPrice] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    ok: bool = False


def _parse_fcfa(raw: str) -> Optional[float]:
    if not raw:
        return None
    match = PRICE_INT_RE.search(raw)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(" ", "").replace(",", ""))
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _fetch_html(url: str = ONCC_URL, timeout: int = 30) -> Optional[str]:
    try:
        import requests

        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )
            },
        )
        if resp.status_code != 200:
            logger.error("ONCC HTTP %s", resp.status_code)
            return None
        return resp.text
    except Exception as exc:
        logger.error("ONCC fetch failed: %s", exc)
        return None


def _extract_by_css(html: str) -> Dict[str, float]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, float] = {}
    mapping = {"arabica": ARABICA_CSS, "robusta": ROBUSTA_CSS}
    for product, selector in mapping.items():
        el = soup.select_one(selector)
        if not el:
            continue
        price = _parse_fcfa(el.get_text())
        if price:
            out[product] = price
    return out


def _extract_by_label(html: str) -> Dict[str, float]:
    """Fallback : cherche Arabica/Robusta puis le nombre le plus proche."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    out: Dict[str, float] = {}
    for product, label in (("arabica", "arabica"), ("robusta", "robusta")):
        pattern = re.compile(
            rf"{label}[^0-9]{{0,80}}([\d\s,]+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            price = _parse_fcfa(match.group(1))
            if price:
                out[product] = price
    return out


def fetch_oncc_coffee_prices(url: str = ONCC_URL) -> OnccCollectionResult:
    """Collecte Arabica + Robusta depuis ONCC."""
    result = OnccCollectionResult()
    today = datetime.now().strftime("%Y-%m-%d")

    html = _fetch_html(url)
    result.attempts.append({"strategy": "http_fetch", "ok": html is not None})
    if not html:
        return result

    css_prices = _extract_by_css(html)
    result.attempts.append(
        {"strategy": "css_selectors", "ok": bool(css_prices), "found": list(css_prices)}
    )

    prices = dict(css_prices)
    strategy = "css_selectors" if css_prices else ""

    if len(prices) < 2:
        label_prices = _extract_by_label(html)
        result.attempts.append(
            {"strategy": "label_fallback", "ok": bool(label_prices), "found": list(label_prices)}
        )
        for k, v in label_prices.items():
            prices.setdefault(k, v)
        if label_prices and not strategy:
            strategy = "label_fallback"

    for product, price in prices.items():
        if product not in ("arabica", "robusta"):
            continue
        if not (100 <= price <= 50000):
            logger.warning("ONCC %s prix hors bornes: %s", product, price)
            continue
        result.prices.append(
            OnccPrice(
                product=product,
                price=price,
                date=today,
                strategy=strategy or "unknown",
            )
        )

    result.ok = len(result.prices) > 0
    return result


def write_collection_journal(
    payload: Dict[str, Any],
    logs_dir: Optional[Path] = None,
) -> Path:
    base = logs_dir or Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    path = base / f"oncc_collection_{day}.json"
    existing: List[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []
    existing.append(payload)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
