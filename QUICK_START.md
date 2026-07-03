# 🚀 Démarrage Rapide - Système de Prédiction Cocoa

## ⚡ En 5 minutes

### 1. Prérequis
- ✅ Docker Desktop installé et démarré
- ✅ Node.js 18+ installé
- ✅ Fichier `.env` configuré

### 2. Démarrer l'API

```bash
# Démarrer Docker
docker-compose up -d

# Vérifier que l'API fonctionne
curl http://localhost:8000/health
```

### 3. Démarrer le Dashboard

```bash
# Dans un nouveau terminal
cd frontend
npm install
npm run dev -- -p 3001
```

### 4. Accéder au système

- **Dashboard Premium**: http://localhost:3001
- **API Documentation**: http://localhost:8000/docs
- **Dashboard Streamlit**: http://localhost:8501

---

## 🔄 Mise à jour quotidienne

### Méthode automatique (Recommandée)

```bash
# Double-clic sur le fichier
update_system.bat
```

### Méthode manuelle

```bash
# 1. Collecter le prix
docker-compose exec -T api python /app/collect_latest_price.py

# 2. Collecter les news
docker-compose exec -T api python /app/collect_news.py

# 3. Réentraîner le modèle
docker-compose exec -T api python /app/train_hybrid_improved.py

# 4. Redémarrer l'API
docker-compose restart api
```

---

## 🧪 Tester l'API

### PowerShell

```powershell
$headers = @{
    "Authorization" = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InVzZXIiLCJleHAiOjE3NzkyMzk1NzYsImlhdCI6MTc3ODYzNDc3NiwidHlwZSI6ImFjY2VzcyJ9.LYT7a8iLQr2xVXq7Uhl6079TnHK58SpnFmzTNU_W50w"
    "Content-Type" = "application/json"
}

$body = @{
    market = "ICE_NY"
    horizons = @(1, 7, 30)
    include_sentiment = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Headers $headers -Body $body
```

### Curl

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InVzZXIiLCJleHAiOjE3NzkyMzk1NzYsImlhdCI6MTc3ODYzNDc3NiwidHlwZSI6ImFjY2VzcyJ9.LYT7a8iLQr2xVXq7Uhl6079TnHK58SpnFmzTNU_W50w" \
  -H "Content-Type: application/json" \
  -d '{
    "market": "ICE_NY",
    "horizons": [1, 7, 30],
    "include_sentiment": true
  }'
```

---

## 🔑 Générer de nouveaux tokens JWT

```bash
docker-compose exec -T api python /app/generate_jwt_token.py
```

Tokens disponibles:
- **Utilisateur** (1h): Prédictions + Performance
- **Admin** (1h): Tous les endpoints + Réentraînement
- **Longue durée** (24h): Développement
- **Test** (7 jours): Tests automatisés

---

## 📊 Commandes utiles

### Docker

```bash
# Voir les logs
docker-compose logs api --tail=100

# Redémarrer l'API
docker-compose restart api

# Arrêter tous les services
docker-compose down

# Reconstruire l'image
docker-compose build
```

### Base de données

```bash
# Se connecter à Supabase
# Aller sur https://supabase.com/dashboard

# Voir les prix récents
SELECT * FROM cocoa_prices ORDER BY date DESC LIMIT 10;

# Voir les news récentes
SELECT title, sentiment_score FROM news_articles ORDER BY published_at DESC LIMIT 10;
```

---

## 🆘 Problèmes courants

### API ne démarre pas
```bash
# Vérifier les logs
docker-compose logs api

# Vérifier les variables d'environnement
cat .env
```

### Token expiré
```bash
# Générer un nouveau token
docker-compose exec -T api python /app/generate_jwt_token.py
```

### Sentiment NULL
```bash
# Recollecte des news
docker-compose exec -T api python /app/collect_news.py
```

### Dashboard ne démarre pas
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev -- -p 3001
```

---

## 📚 Documentation complète

Pour plus de détails, consultez:
- **README_COMPLET.md**: Documentation exhaustive
- **CHANGELOG_MAI_2026.md**: Historique des changements
- **JWT_AUTHENTICATION_GUIDE.md**: Guide d'authentification
- **DASHBOARD_GUIDE.md**: Guide du dashboard

---

## 🎯 Prochaines étapes

1. ✅ Système démarré
2. ⏭️ Tester les prédictions
3. ⏭️ Configurer la mise à jour automatique quotidienne
4. ⏭️ Personnaliser le dashboard
5. ⏭️ Monitorer les performances

---

**Besoin d'aide ?** Consultez la section Dépannage dans README_COMPLET.md
