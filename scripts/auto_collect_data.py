"""
Script d'extraction automatique des donnees cacao ICE London.
Remplace l'ancienne collecte Yahoo Finance (CC=F).
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.data_collection.ice_london_collector import fetch_ice_london_spot
from src.models.market_registry import get_market_config


def collect_and_save_data():
    market = get_market_config("cocoa")
    ice_url = getattr(market, "ice_london_url", None) or (
        "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
    )

    logger.info("=" * 80)
    logger.info("EXTRACTION AUTOMATIQUE CACAO ICE LONDON")
    logger.info("=" * 80)

    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )
        logger.info("Connexion Supabase etablie")
    except Exception as e:
        logger.error("Erreur connexion Supabase: %s", e)
        return False

    try:
        response = (
            supabase.table(market.price_table)
            .select("date, price")
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            logger.info(
                "Dernier prix en base: %s %s le %s",
                response.data[0]["price"],
                market.unit,
                response.data[0]["date"],
            )
    except Exception as e:
        logger.error("Erreur lecture base: %s", e)
        return False

    result = fetch_ice_london_spot(url=ice_url, price_bounds=market.price_bounds)
    if result is None:
        logger.error("Collecte ICE London echouee")
        return False

    lo, hi = market.price_bounds
    if not (lo <= result.price <= hi):
        logger.error("Prix %s hors bornes (%s-%s)", result.price, lo, hi)
        return False

    check = (
        supabase.table(market.price_table)
        .select("id")
        .eq("date", result.date)
        .execute()
    )
    if check.data:
        logger.info("Date %s deja en base", result.date)
        return True

    try:
        supabase.table(market.price_table).insert(
            {
                "date": result.date,
                "price": float(result.price),
                "symbol": result.symbol,
                "source": result.source,
                "collected_at": datetime.now().isoformat(),
            }
        ).execute()
        logger.info("Insere: %s - %s %s", result.date, result.price, market.unit)
        return True
    except Exception as e:
        logger.error("Erreur insertion: %s", e)
        return False


if __name__ == "__main__":
    success = collect_and_save_data()
    sys.exit(0 if success else 1)
