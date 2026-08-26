"""
Script pour générer des tokens JWT pour l'API de prédiction.

Ce script permet de créer des tokens d'accès pour les utilisateurs et les administrateurs.
"""

import sys
from datetime import timedelta
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.auth import create_user_token, create_access_token, create_service_token
from config.settings import get_settings

print("=" * 80)
print("GÉNÉRATEUR DE TOKENS JWT")
print("=" * 80)

# Vérifier que la configuration est chargée
try:
    settings = get_settings()
    print(f"\n✅ Configuration chargée")
    print(f"   Secret Key: {settings.secret_key[:20]}...")
    print(f"   Algorithm: {settings.jwt_algorithm}")
    print(f"   Expiration: {settings.access_token_expire_minutes} minutes")
except Exception as e:
    print(f"\n❌ Erreur lors du chargement de la configuration: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("TOKENS GÉNÉRÉS")
print("=" * 80)

# ============================================================================
# 1. TOKEN UTILISATEUR STANDARD (1 heure)
# ============================================================================
print("\n[1] Token Utilisateur Standard (1 heure)")
print("-" * 80)

user_token = create_user_token(
    user_id="trader@example.com",
    role="user"
)

print(f"\nUser ID: trader@example.com")
print(f"Role: user")
print(f"Expiration: 1 heure")
print(f"\nToken:")
print(user_token)

# ============================================================================
# 2. TOKEN ADMINISTRATEUR (1 heure)
# ============================================================================
print("\n" + "=" * 80)
print("[2] Token Administrateur (1 heure)")
print("-" * 80)

admin_token = create_user_token(
    user_id="admin@example.com",
    role="admin"
)

print(f"\nUser ID: admin@example.com")
print(f"Role: admin")
print(f"Expiration: 1 heure")
print(f"\nToken:")
print(admin_token)

# ============================================================================
# 3. TOKEN SERVICE (1 an) — batch + BFF Next.js
# ============================================================================
print("\n" + "=" * 80)
print("[3] Token Service (1 an) — API_TOKEN pour .env / Vercel")
print("-" * 80)

service_token = create_service_token(
    user_id="service@dashboard",
    role="user",
    days=365,
)

print(f"\nUser ID: service@dashboard")
print(f"Role: user")
print(f"Expiration: 365 jours")
print(f"\nToken (à mettre dans API_TOKEN=...):")
print(service_token)

# ============================================================================
# 4. TOKEN LONGUE DURÉE (24 heures)
# ============================================================================
print("\n" + "=" * 80)
print("[4] Token Longue Durée (24 heures)")
print("-" * 80)

long_token = create_access_token(
    data={"sub": "developer@example.com", "role": "user"},
    expires_delta=timedelta(hours=24)
)

print(f"\nUser ID: developer@example.com")
print(f"Role: user")
print(f"Expiration: 24 heures")
print(f"\nToken:")
print(long_token)

# ============================================================================
# 5. TOKEN DE TEST (7 jours)
# ============================================================================
print("\n" + "=" * 80)
print("[5] Token de Test (7 jours)")
print("-" * 80)

test_token = create_access_token(
    data={"sub": "test@example.com", "role": "user"},
    expires_delta=timedelta(days=7)
)

print(f"\nUser ID: test@example.com")
print(f"Role: user")
print(f"Expiration: 7 jours")
print(f"\nToken:")
print(test_token)

# ============================================================================
# INSTRUCTIONS D'UTILISATION
# ============================================================================
print("\n" + "=" * 80)
print("COMMENT UTILISER CES TOKENS")
print("=" * 80)

print("""
1. Stockez le token service dans .env (JAMAIS dans git) :

   API_TOKEN=<token service ci-dessus>

2. Frontend Vercel : variable server-only API_TOKEN (pas NEXT_PUBLIC_*).

3. AVEC CURL :

   curl -X POST "http://localhost:8000/api/v1/predict" \\
     -H "Authorization: Bearer $API_TOKEN" \\
     -H "Content-Type: application/json" \\
     -d '{"market": "ICE_NY", "horizons": [1, 7, 30], "include_sentiment": true}'
""")

print("\n" + "=" * 80)
print("✅ Tokens générés avec succès !")
print("=" * 80)
