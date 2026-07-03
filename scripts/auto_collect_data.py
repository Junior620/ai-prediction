"""
Script d'extraction automatique des données
Collecte les prix du cacao depuis Yahoo Finance et les insère dans Supabase
"""

import yfinance as yf
from datetime import datetime, timedelta
from supabase import create_client
import os
from dotenv import load_dotenv
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def collect_and_save_data():
    """Collecte les données et les sauvegarde dans Supabase"""
    
    logger.info("=" * 80)
    logger.info("EXTRACTION AUTOMATIQUE DES DONNÉES")
    logger.info("=" * 80)
    
    # Connexion Supabase
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        logger.info("✅ Connexion Supabase établie")
    except Exception as e:
        logger.error(f"❌ Erreur connexion Supabase: {e}")
        return False
    
    # Vérifier le dernier prix dans la base
    try:
        response = supabase.table("cocoa_prices").select("date, price").order("date", desc=True).limit(1).execute()
        
        if response.data:
            last_date = response.data[0]['date']
            last_price = response.data[0]['price']
            logger.info(f"Dernier prix en base: ${last_price:,.2f} le {last_date}")
        else:
            logger.warning("Aucune donnée dans la base")
            last_date = None
    except Exception as e:
        logger.error(f"❌ Erreur lecture base: {e}")
        return False
    
    # Télécharger les données récentes depuis Yahoo Finance
    logger.info("Téléchargement des données depuis Yahoo Finance...")
    
    ticker = yf.Ticker("CC=F")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning("Aucune donnée récente disponible")
            return False
        
        logger.info(f"✅ {len(df)} points téléchargés")
        
        # Insérer les nouvelles données
        inserted = 0
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            price = float(row['Close'])
            
            # Vérifier si la date existe déjà
            check = supabase.table("cocoa_prices").select("id").eq("date", date_str).execute()
            
            if not check.data:
                try:
                    supabase.table("cocoa_prices").insert({
                        "date": date_str,
                        "price": price,
                        "symbol": "CC=F",
                        "source": "yahoo_finance",
                        "collected_at": datetime.now().isoformat()
                    }).execute()
                    inserted += 1
                    logger.info(f"   ✅ Inséré: {date_str} - ${price:,.2f}")
                except Exception as e:
                    logger.error(f"   ❌ Erreur insertion {date_str}: {e}")
        
        logger.info(f"✅ {inserted} nouvelles données insérées")
        return inserted > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement: {e}")
        return False

if __name__ == "__main__":
    success = collect_and_save_data()
    exit(0 if success else 1)
