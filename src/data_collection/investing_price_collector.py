"""
Collecteur de prix Investing.com (Selenium headless), paramétrable par instrument.

Utilisé pour le café robusta (page london-coffee). Le symbole du contrat actif
(ex. RCU6) est détecté depuis le titre de la page pour suivre les rollovers.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

PRICE_SELECTORS = [
    (By.CSS_SELECTOR, "[data-test='instrument-price-last']"),
    (By.CLASS_NAME, "instrument-price_last__KQzyA"),
    (By.CSS_SELECTOR, ".text-5xl"),
]

# Titre type : "London Coffee (RCU6)"
SYMBOL_PATTERN = re.compile(r"\(([A-Z]{2,4}[A-Z0-9]{1,4})\)")


def fetch_investing_price(
    url: str,
    fallback_symbol: str = "",
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Scrape le dernier prix affiché sur une page instrument Investing.com.

    Returns:
        {"price": float, "symbol": str, "date": "YYYY-MM-DD", "source": "investing_com"}
        ou None en cas d'échec.
    """
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options,
        )
        driver.get(url)
        wait = WebDriverWait(driver, timeout)

        price: Optional[float] = None
        for by, selector in PRICE_SELECTORS:
            try:
                element = wait.until(EC.presence_of_element_located((by, selector)))
                text = element.text.strip()
                if text:
                    price = float(text.replace(",", ""))
                    break
            except Exception:
                continue

        if price is None:
            logger.error("Investing.com: prix introuvable sur %s", url)
            return None

        # Symbole du contrat actif depuis le titre (suit les rollovers)
        symbol = fallback_symbol
        match = SYMBOL_PATTERN.search(driver.title or "")
        if match:
            symbol = match.group(1)

        return {
            "price": price,
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "investing_com",
        }

    except Exception as exc:
        logger.error("Investing.com scrape failed (%s): %s", url, exc)
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
