"""
Collecteur prix cacao ICE London (£/T) — Playwright + fallbacks.

Strategies (ordre) :
1. Playwright + XPath / CSS sur la page ICE officielle
2. Parse tableau HTML (plusieurs lignes pour backfill)
3. Regex settlement/last sur HTML rendu
4. Fallback Investing.com UK cocoa
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ICE_URL = (
    "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
)
INVESTING_FALLBACK_URL = "https://uk.investing.com/commodities/london-cocoa"

ICE_XPATH = (
    "/html/body/div[1]/div/main/div/div/div/div/div/div[4]/div/div/div[1]/"
    "table/tbody[1]/tr[1]/td[2]"
)
ICE_CSS_FIRST_CELL = "table tbody tr:first-child td:nth-child(2)"

PRICE_RE = re.compile(r"[\d,]+\.?\d*")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}\s+\w+\s+\d{4})")
SETTLEMENT_RE = re.compile(
    r"(?:settlement|last|close|closing)[^0-9]{0,40}([\d,]+\.?\d*)",
    re.IGNORECASE,
)
TABLE_ROW_RE = re.compile(
    r"<tr[^>]*>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class IceLondonResult:
    price: float
    date: str
    source: str
    symbol: str = "LCC"
    strategy: str = ""
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def _parse_price(raw: str) -> Optional[float]:
    if not raw:
        return None
    match = PRICE_RE.search(str(raw).replace(" ", ""))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_date(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw[:10] if fmt != "%d/%m/%Y" else raw, fmt).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            continue
    return None


def parse_table_rows_from_html(html: str, max_rows: int = 60) -> List[Dict[str, Any]]:
    """Extrait date + settlement depuis un tableau ICE rendu."""
    rows: List[Dict[str, Any]] = []
    if not html:
        return rows

    for match in TABLE_ROW_RE.finditer(html):
        date_raw = _strip_tags(match.group(1))
        price_raw = _strip_tags(match.group(2))
        price = _parse_price(price_raw)
        date_str = _normalize_date(date_raw)
        if price is None:
            continue
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        rows.append(
            {
                "date": date_str,
                "price": price,
                "source": "ice_london",
                "symbol": "LCC",
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def extract_price_regex(html: str) -> Optional[float]:
    if not html:
        return None
    match = SETTLEMENT_RE.search(html)
    if match:
        return _parse_price(match.group(1))
    # Dernier recours : premier nombre plausible dans la zone tableau
    for m in PRICE_RE.finditer(html):
        val = _parse_price(m.group(0))
        if val and 800 <= val <= 15000:
            return val
    return None


def _scrape_playwright(
    url: str, timeout_ms: int = 45000
) -> Tuple[Optional[str], str, Optional[float]]:
    """Retourne (html, strategie, prix_direct) ou (None, erreur, None)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return None, f"playwright_missing:{exc}", None

    html = None
    strategy = ""
    direct_price: Optional[float] = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_selector("table tbody tr td", timeout=15000)

            # 1. XPath utilisateur
            try:
                loc = page.locator(f"xpath={ICE_XPATH}").first
                text = loc.text_content(timeout=5000)
                price = _parse_price(text or "")
                if price:
                    strategy = "playwright_xpath"
                    direct_price = price
                    html = page.content()
                    browser.close()
                    return html, strategy, direct_price
            except Exception:
                pass

            # 2. CSS fallback
            try:
                loc = page.locator(ICE_CSS_FIRST_CELL).first
                text = loc.text_content(timeout=5000)
                price = _parse_price(text or "")
                if price:
                    strategy = "playwright_css"
                    direct_price = price
                    html = page.content()
                    browser.close()
                    return html, strategy, direct_price
            except Exception:
                pass

            html = page.content()
            strategy = "playwright_html"
            browser.close()
    except Exception as exc:
        return None, f"playwright_error:{exc}", None

    return html, strategy, direct_price


def _investing_fallback() -> Optional[IceLondonResult]:
    try:
        from src.data_collection.investing_price_collector import fetch_investing_price
    except ImportError:
        return None

    data = fetch_investing_price(url=INVESTING_FALLBACK_URL, fallback_symbol="LCC")
    if not data:
        return None
    return IceLondonResult(
        price=float(data["price"]),
        date=data.get("date") or datetime.now().strftime("%Y-%m-%d"),
        source="ice_london_fallback",
        symbol=data.get("symbol") or "LCC",
        strategy="investing_fallback",
    )


def fetch_ice_london_spot(
    url: str = DEFAULT_ICE_URL,
    price_bounds: Tuple[float, float] = (800.0, 15000.0),
) -> Optional[IceLondonResult]:
    """
    Collecte le prix spot du jour avec chaine de fallbacks.
    Retourne None si toutes les strategies echouent.
    """
    attempts: List[Dict[str, Any]] = []
    lo, hi = price_bounds

    # Playwright
    t0 = time.time()
    html, pw_info, direct_price = _scrape_playwright(url)
    attempts.append(
        {"strategy": "playwright", "ok": html is not None, "detail": pw_info, "ms": int((time.time() - t0) * 1000)}
    )

    if direct_price and lo <= direct_price <= hi:
        return IceLondonResult(
            price=direct_price,
            date=datetime.now().strftime("%Y-%m-%d"),
            source="ice_london",
            strategy=pw_info or "playwright_direct",
            attempts=attempts,
        )

    if html:
        rows = parse_table_rows_from_html(html, max_rows=1)
        if rows:
            row = rows[0]
            price = float(row["price"])
            if lo <= price <= hi:
                return IceLondonResult(
                    price=price,
                    date=row["date"],
                    source="ice_london",
                    strategy=pw_info or "playwright_table",
                    attempts=attempts,
                )

        price = extract_price_regex(html)
        if price and lo <= price <= hi:
            return IceLondonResult(
                price=price,
                date=datetime.now().strftime("%Y-%m-%d"),
                source="ice_london",
                strategy=pw_info or "playwright_regex",
                attempts=attempts,
            )

    # Investing fallback
    t1 = time.time()
    fb = _investing_fallback()
    attempts.append(
        {
            "strategy": "investing_fallback",
            "ok": fb is not None,
            "ms": int((time.time() - t1) * 1000),
        }
    )
    if fb and lo <= fb.price <= hi:
        fb.attempts = attempts
        return fb

    logger.error("ICE London: toutes les strategies ont echoue")
    return None


def fetch_ice_london_history(
    url: str = DEFAULT_ICE_URL,
    max_rows: int = 120,
    price_bounds: Tuple[float, float] = (800.0, 15000.0),
) -> List[Dict[str, Any]]:
    """Scrape N lignes du tableau ICE (si historique journalier disponible)."""
    lo, hi = price_bounds
    rows = _scrape_playwright_table_rows(url, max_rows=max_rows)
    if not rows:
        html, _, _ = _scrape_playwright(url)
        if html:
            rows = parse_table_rows_from_html(html, max_rows=max_rows)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        price = float(row["price"])
        if not (lo <= price <= hi):
            continue
        d = row["date"]
        if d in seen:
            continue
        seen.add(d)
        out.append(row)
    return out


def _scrape_playwright_table_rows(
    url: str,
    max_rows: int = 120,
    timeout_ms: int = 45000,
) -> List[Dict[str, Any]]:
    """Lit les lignes d'un tableau ICE via Playwright (date + settlement)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_selector("table tbody tr", timeout=15000)
            tr = page.locator("table tbody tr")
            count = min(tr.count(), max_rows)
            for i in range(count):
                cells = tr.nth(i).locator("td")
                n = cells.count()
                if n < 2:
                    continue
                texts = [cells.nth(j).inner_text().strip() for j in range(n)]
                # Historique journalier : col0=date, col1=settlement
                date_str = _normalize_date(texts[0])
                price = _parse_price(texts[1])
                if not date_str or price is None:
                    # Courbe contrats (SEP26, ...) — pas un backfill journalier
                    if re.match(r"^[A-Z]{3}\d{2}$", texts[0].upper()):
                        continue
                    # Essayer date dans une autre colonne
                    for t in texts[1:]:
                        d = _normalize_date(t.split("\n")[0])
                        if d:
                            date_str = d
                            break
                    if not date_str:
                        continue
                rows.append(
                    {
                        "date": date_str,
                        "price": price,
                        "source": "ice_london",
                        "symbol": "LCC",
                    }
                )
            browser.close()
    except Exception as exc:
        logger.warning("Playwright table rows failed: %s", exc)
    return rows


def bootstrap_london_from_ny_prices(
    ny_rows: List[Dict[str, Any]],
    fx_by_day: Dict[str, float],
    default_fx: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Convertit cocoa_prices (ICE NY, USD/T) en serie GBP/T approximative
    via taux USD/GBP journaliers (Frankfurter).
    """
    out: List[Dict[str, Any]] = []
    last_fx = default_fx
    by_day: Dict[str, Dict[str, Any]] = {}
    for row in sorted(ny_rows, key=lambda r: r["date"]):
        day = str(row["date"])[:10]
        fx = fx_by_day.get(day, last_fx)
        if day in fx_by_day:
            last_fx = fx
        price_usd = float(row["price"])
        price_gbp = round(price_usd * fx, 2)
        by_day[day] = {
            "date": day,
            "price": price_gbp,
            "symbol": "LCC",
            "source": "ice_london_bootstrap_fx",
        }
    out = list(by_day.values())
    return out


def fetch_usd_gbp_rates(start_date: str, end_date: str) -> Dict[str, float]:
    """Taux GBP pour 1 USD par jour (Frankfurter / BCE), requetes par annee."""
    import requests
    from datetime import date

    rates: Dict[str, float] = {}
    start_y = int(start_date[:4])
    end_y = int(end_date[:4])
    for year in range(start_y, end_y + 1):
        y_start = f"{year}-01-01" if year > start_y else start_date
        y_end = f"{year}-12-31" if year < end_y else end_date
        try:
            url = f"https://api.frankfurter.app/{y_start}..{y_end}?from=USD&to=GBP"
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                continue
            for day, cur in (resp.json().get("rates") or {}).items():
                gbp = cur.get("GBP")
                if gbp:
                    rates[day] = float(gbp)
        except Exception as exc:
            logger.warning("Frankfurter FX %s: %s", year, exc)
    return rates


def write_collection_journal(
    payload: Dict[str, Any],
    logs_dir: Optional[Path] = None,
) -> Path:
    """Ecrit logs/ice_london_collection_YYYYMMDD.json."""
    base = logs_dir or Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    path = base / f"ice_london_collection_{day}.json"
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
