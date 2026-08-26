"""
Collecteur des contrats a terme Investing.com (table BarchartDataTable).

Page type :
  https://fr.investing.com/commodities/us-cocoa-contracts
  https://www.investing.com/commodities/london-coffee-contracts  (robusta, futur)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TABLE_PATTERN = re.compile(
    r'<table[^>]*id="BarchartDataTable"[^>]*>.*?</table>',
    re.IGNORECASE | re.DOTALL,
)
ROW_PATTERN = re.compile(
    r'<tr[^>]*id="pair_\d+"[^>]*>(.*?)</tr>',
    re.IGNORECASE | re.DOTALL,
)
CELL_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
SYMBOL_PATTERN = re.compile(r"symbol=([A-Z0-9]+)", re.IGNORECASE)


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return cleaned.replace("&nbsp;", " ").strip()


def _parse_number(raw: str) -> Optional[float]:
    if not raw:
        return None
    value = raw.strip().rstrip("sS")  # prix reglement (5819s)
    value = value.replace(",", "").replace(" ", "")
    if not value or value in ("--", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_change(raw: str) -> Optional[float]:
    if not raw:
        return None
    value = _strip_html(raw).replace(",", "").replace(" ", "")
    if not value or value in ("--", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_investing_futures_table(html: str) -> List[Dict[str, Any]]:
    """
    Parse la table Contrats (Mois, Dernier, Var., Ouverture, + Haut, + Bas, Volume, Heure).

    Returns list of dicts compatible with cocoa_futures JSONB:
      contract, symbol, price_usd (+ champs optionnels)
    """
    match = TABLE_PATTERN.search(html or "")
    if not match:
        logger.error("Investing.com: table BarchartDataTable introuvable")
        return []

    table_html = match.group(0)
    contracts: List[Dict[str, Any]] = []

    for row_html in ROW_PATTERN.findall(table_html):
        cells = CELL_PATTERN.findall(row_html)
        # icon | mois | dernier | var | open | high | low | volume | heure | graph
        if len(cells) < 9:
            continue

        month_cell = cells[1]
        sym_match = SYMBOL_PATTERN.search(month_cell)
        symbol = sym_match.group(1).upper() if sym_match else ""
        contract = _strip_html(month_cell)
        if not contract:
            continue

        last_raw = _strip_html(cells[2])
        price = _parse_number(last_raw)
        if price is None:
            continue

        entry: Dict[str, Any] = {
            "contract": contract,
            "symbol": symbol or contract,
            "price_usd": price,
            "settlement": last_raw.lower().endswith("s"),
        }

        change = _parse_change(cells[3])
        if change is not None:
            entry["change"] = change
        for key, idx in (("open", 4), ("high", 5), ("low", 6)):
            val = _parse_number(_strip_html(cells[idx]))
            if val is not None:
                entry[key] = val

        vol = _parse_number(_strip_html(cells[7]))
        if vol is not None:
            entry["volume"] = int(vol)

        quote_time = _strip_html(cells[8])
        if quote_time:
            entry["quote_time"] = quote_time

        contracts.append(entry)

    return contracts


def fetch_investing_futures(
    url: str,
    timeout: int = 30,
) -> Optional[List[Dict[str, Any]]]:
    """Telecharge et parse la courbe contrats Investing.com."""
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError as exc:
        logger.error("curl_cffi manquant (%s). Installez: pip install curl_cffi", exc)
        return None

    try:
        response = cffi_requests.get(
            url,
            impersonate="chrome131",
            timeout=timeout,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            },
        )
    except Exception as exc:
        logger.error("Investing.com futures HTTP failed (%s): %s", url, exc)
        return None

    if response.status_code != 200:
        logger.error("Investing.com futures HTTP %s sur %s", response.status_code, url)
        return None

    html = response.text or ""
    if "Just a moment" in html or "cf-challenge" in html.lower():
        logger.error("Investing.com futures: challenge Cloudflare non resolu (%s)", url)
        return None

    contracts = parse_investing_futures_table(html)
    if not contracts:
        logger.error("Investing.com futures: aucun contrat parse sur %s", url)
        return None

    logger.info("Investing.com futures OK: %d contrats depuis %s", len(contracts), url)
    return contracts
