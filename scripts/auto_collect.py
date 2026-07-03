"""
Script d'extraction automatique des données
Collecte les prix du cacao et les insère dans Supabase
"""

import subprocess
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_collect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_data():
    """Collecte les données via Docker"""
    
    logger.info("=" * 80)
    logger.info("COLLECTE AUTOMATIQUE DES DONNÉES")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        # Exécuter le script de collecte dans Docker
        logger.info("Lancement de la collecte dans Docker...")
        result = subprocess.run(
            ["docker", "exec", "cocoa-api", "python", "/app/collect_latest_price.py"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ Collecte terminée avec succès")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Erreur collecte: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout: la collecte a pris trop de temps")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = collect_data()
    exit(0 if success else 1)
