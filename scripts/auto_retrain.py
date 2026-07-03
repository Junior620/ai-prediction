"""
Script d'entraînement automatique du modèle
Réentraîne le modèle avec les dernières données et redémarre l'API
"""

import subprocess
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retrain_model():
    """Réentraîne le modèle dans Docker"""
    
    logger.info("=" * 80)
    logger.info("RÉENTRAÎNEMENT AUTOMATIQUE DU MODÈLE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        # Exécuter le script d'entraînement dans Docker
        logger.info("Lancement de l'entraînement dans Docker...")
        result = subprocess.run(
            ["docker", "exec", "cocoa-api", "python", "/app/train_hybrid_improved.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ Entraînement terminé avec succès")
            logger.info(result.stdout)
            
            # Redémarrer l'API pour charger les nouveaux modèles
            logger.info("Redémarrage de l'API...")
            restart_result = subprocess.run(
                ["docker-compose", "restart", "api"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if restart_result.returncode == 0:
                logger.info("✅ API redémarrée avec succès")
                return True
            else:
                logger.error(f"❌ Erreur redémarrage API: {restart_result.stderr}")
                return False
        else:
            logger.error(f"❌ Erreur entraînement: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout: l'entraînement a pris trop de temps")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = retrain_model()
    exit(0 if success else 1)
