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
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Titre / H1 type : "London Coffee (RCU6)"
SYMBOL_PATTERN = re.compile(r"\(([A-Z]{2,4}[A-Z0-9]{1,4})\)")

_CLOSE_OBJECT = re.compile(r"\{[^{}]*last_close_date[^{}]*\}")
_CLOSE_DATE = re.compile(r'"last_close_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
_CLOSE_PRICE = re.compile(
    r'"(?:last_close|previousClose|close)"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
)
_OBJECT_SYMBOL = re.compile(r'"(?:symbol|ticker)"\s*:\s*"([A-Z]{2,4}[A-Z0-9]{1,4})"')
_LAST_PRICE = re.compile(
    r'data-test=["\']instrument-price-last["\'][^>]*>\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)',
    re.IGNORECASE,
)
_MARKED_DATE = re.compile(
    r'data-test=["\']instrument-session-date["\'][^>]*>\s*(\d{4}-\d{2}-\d{2})',
    re.IGNORECASE,
)
_HEADER_STATE = re.compile(
    r'data-test=["\']trading-state-label["\'][^>]*>\s*([^<]+)',
    re.IGNORECASE,
)
_LONDON = ZoneInfo("Europe/London")


def _real_session_day(value: str) -> Optional[str]:
    text = str(value or "")[:10]
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    if parsed.strftime("%Y-%m-%d") != text:
        return None
    return text


def _strip_related(html: str) -> str:
    """Drop related objects, including ones nested inside another object."""
    out: list[str] = []
    cursor = 0
    lowered = (html or "").lower()
    while True:
        found = lowered.find('"related"', cursor)
        if found < 0:
            out.append(html[cursor:])
            break
        brace = html.find("{", found)
        if brace < 0:
            out.append(html[cursor:])
            break
        depth = 0
        end = None
        for index in range(brace, len(html)):
            char = html[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        out.append(html[cursor:found])
        if end is None:
            break
        cursor = end
    return "".join(out)


def _without_other_instruments(html: str, symbol: str) -> str:
    """A date that belongs to another instrument cannot date this contract."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    for node in list(soup.select("[data-instrument]")):
        name = str(node.get("data-instrument") or "").strip()
        if symbol and name.upper() == symbol.upper():
            continue
        node.decompose()
    return str(soup)


def _close_records(html: str) -> list[dict[str, Any]]:
    """Close price, close date and contract from the same object."""
    cleaned = _strip_related(html or "")
    found = []
    for match in _CLOSE_OBJECT.finditer(cleaned):
        blob = match.group(0)
        day = _CLOSE_DATE.search(blob)
        price = _CLOSE_PRICE.search(blob)
        symbol = _OBJECT_SYMBOL.search(blob)
        if not day or not price or symbol is None:
            continue
        session = _real_session_day(day.group(1))
        amount = _parse_price(price.group(1))
        if not session or amount is None:
            continue
        found.append(
            {
                "price": amount,
                "session_date": session,
                "symbol": symbol.group(1),
                "price_kind": "session_close",
            }
        )
    return found


def _london_session_day(stamp: str) -> Optional[str]:
    """Calendar day in London of a timestamp that already carries a timezone."""
    text = str(stamp or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(_LONDON).date().isoformat()


def _header_clocks(window: str) -> list[str]:
    found = []
    for tag in re.finditer(r"<time\b[^>]*>", window, re.IGNORECASE):
        tag_text = tag.group(0)
        if re.search(r'data-test=["\']trading-time-label["\']', tag_text, re.IGNORECASE) is None:
            continue
        stamp = re.search(r'dateTime=["\']([^"\']+)["\']', tag_text, re.IGNORECASE)
        if stamp:
            found.append(stamp.group(1))
    return found


def _closed_header_quote(html: str, symbol: str) -> Optional[dict[str, Any]]:
    """Last price dated by the header clock, only once the session is closed."""
    prices = list(_LAST_PRICE.finditer(html or ""))
    if len(prices) != 1 or '"last_close"' in (html or ""):
        return None
    window = html[prices[0].start() : prices[0].start() + 2500]
    states = [item.strip().casefold() for item in _HEADER_STATE.findall(window)]
    clocks = _header_clocks(window)
    if states != ["closed"]:
        return None
    if len(clocks) != 1:
        return None
    session = _london_session_day(clocks[0])
    amount = _parse_price(prices[0].group(1))
    if session is None or amount is None:
        return None
    return {
        "price": amount,
        "symbol": symbol,
        "session_date": session,
        "price_kind": "closed_header",
        "quarantine_reason": None,
    }


def linked_quote(html: str, fallback_symbol: str = "") -> dict[str, Any]:
    """One contract, one price, one session, and the kind of that price.

    A close date is kept only with the close price of the same object.
    A date found elsewhere does not date the last traded price.
    """
    page = _strip_related(html or "")
    symbol = _extract_symbol(page, fallback_symbol=fallback_symbol)
    closes = [
        item
        for item in _close_records(page)
        if item["symbol"] in (None, "", symbol) or item["symbol"] == symbol
    ]
    if symbol:
        named = [item for item in closes if item["symbol"] == symbol]
        if named:
            closes = named
    if len(closes) == 1:
        chosen = closes[0]
        return {
            "price": chosen["price"],
            "symbol": chosen["symbol"] or symbol,
            "session_date": chosen["session_date"],
            "price_kind": "session_close",
            "quarantine_reason": None,
        }
    owned = _without_other_instruments(page, symbol)
    last_prices = _LAST_PRICE.findall(owned)
    marked = [day for day in (_real_session_day(item) for item in _MARKED_DATE.findall(owned)) if day]
    if len(last_prices) == 1 and len(marked) == 1 and not closes and '"last_close"' not in page:
        amount = _parse_price(last_prices[0])
        if amount is not None:
            return {
                "price": amount,
                "symbol": symbol,
                "session_date": marked[0],
                "price_kind": "marked_session",
                "quarantine_reason": None,
            }
    closed = _closed_header_quote(owned, symbol)
    if closed is not None:
        return closed
    amount = _parse_price(last_prices[0]) if last_prices else _extract_price(page)
    return {
        "price": amount,
        "symbol": symbol,
        "session_date": None,
        "price_kind": "intraday",
        "quarantine_reason": "lien_prix_date_non_prouve",
    }


def session_quote_from_html(html: str) -> tuple[Optional[str], str]:
    """Session and kind when the price and the date are one record."""
    quote = linked_quote(html)
    if quote.get("quarantine_reason"):
        return None, str(quote.get("price_kind") or "intraday")
    return quote.get("session_date"), str(quote.get("price_kind") or "intraday")


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
            impersonate="chrome146",
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

    quote = linked_quote(html, fallback_symbol=fallback_symbol)
    price = quote.get("price")
    if price is None:
        logger.error("Investing.com: prix introuvable sur %s", url)
        return None

    logger.info(
        "Investing.com OK: %s = %.2f (contrat %s, séance %s, %s)",
        url,
        price,
        quote.get("symbol") or "?",
        quote.get("session_date") or "absente",
        quote.get("price_kind"),
    )
    return {
        "price": price,
        "symbol": quote.get("symbol") or fallback_symbol,
        "session_date": quote.get("session_date"),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "source": "investing_com",
        "price_kind": quote.get("price_kind"),
        "quarantine_reason": quote.get("quarantine_reason"),
    }
