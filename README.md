# 🍫 Cocoa Price Prediction System

> **Version 2.0.0** - Système de prédiction des prix du cacao avec ML hybride (Prophet + XGBoost + FinBERT)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Démarrage Rapide

```bash
# 1. Démarrer l'API
docker-compose up -d

# 2. Démarrer le Dashboard
cd frontend && npm install && npm run dev -- -p 3001

# 3. Accéder au système
# Dashboard: http://localhost:3001
# API Docs: http://localhost:8000/docs
```

**📖 [Guide de démarrage complet →](QUICK_START.md)**

---

## 📚 Documentation

- **[README_COMPLET.md](README_COMPLET.md)** - Documentation exhaustive (Architecture, Installation, API, Dépannage)
- **[QUICK_START.md](QUICK_START.md)** - Démarrage en 5 minutes
- **[CHANGELOG_MAI_2026.md](CHANGELOG_MAI_2026.md)** - Historique des changements
- **[JWT_AUTHENTICATION_GUIDE.md](JWT_AUTHENTICATION_GUIDE.md)** - Guide d'authentification
- **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)** - Guide du dashboard

---

## ✨ Fonctionnalités

- 🎯 **Prédictions multi-horizons**: 1, 7, et 30 jours
- 🤖 **Modèle hybride**: Prophet + XGBoost + FinBERT
- 📰 **Analyse de sentiment**: Intégration des news financières
- 🔐 **API REST sécurisée**: Authentification JWT
- 📊 **Dashboards premium**: Next.js (premium) + Streamlit (simple)
- 🐳 **Containerisé**: Docker + Docker Compose
- 🔄 **Mise à jour automatique**: Script batch complet

---

## 📊 Performance

Métriques de **validation walk-forward honnête** (268 origines, déc. 2020 – avr. 2026) :

| Métrique | J+1 | J+7 | J+30 |
|----------|-----|-----|------|
| **MAPE (XGBoost)** | ~10,3 % | ~11,3 % | ~16,1 % |
| **Direction** | ~51 % | ~49 % | ~50 % |

> L'ancien holdout 80/20 (~4,8 % MAPE) surestimait la performance. Référence : `GET /api/v1/validation/metrics` ou `reports/walk_forward/`.

| Données | 1 618 points | 2020-2026 |

---

## 🔄 Mise à Jour Quotidienne

### Automatique (Recommandé)

```bash
# Double-clic sur le fichier
update_system.bat
```

Ce script effectue automatiquement:
1. ✅ Collecte du prix actuel (Yahoo Finance)
2. ✅ Collecte des news + analyse sentiment
3. ✅ Réentraînement du modèle
4. ✅ Mise à jour de l'API
5. ✅ Redémarrage des services

### Manuelle

```bash
docker-compose exec -T api python /app/collect_latest_price.py
docker-compose exec -T api python /app/collect_news.py
docker-compose exec -T api python /app/train_hybrid_improved.py
docker-compose restart api
```

---

## 🏗️ Architecture

```
┌─────────────┐
│  Dashboard  │ ← Next.js Premium (Port 3001)
│  Next.js    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  API        │ ← FastAPI + JWT Auth (Port 8000)
│  FastAPI    │
└──────┬──────┘
       │
   ┌───┴────┬──────────┐
   ▼        ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Supabase│ │ Models │ │ Redis  │
│(Prices │ │Prophet │ │ Cache  │
│ News)  │ │XGBoost │ │(Opt.)  │
└────────┘ │FinBERT │ └────────┘
           └────────┘
```

---

## 🧪 Test de l'API

```powershell
$headers = @{
    "Authorization" = "Bearer VOTRE_TOKEN_ICI"
    "Content-Type" = "application/json"
}

$body = @{
    market = "ICE_NY"
    horizons = @(1, 7, 30)
    include_sentiment = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Headers $headers -Body $body
```

**🔑 [Générer un token JWT →](JWT_AUTHENTICATION_GUIDE.md)**

---

## 📁 Structure du Projet

```
prediction/
├── update_system.bat          # Script de mise à jour automatique
├── README_COMPLET.md          # Documentation complète
├── QUICK_START.md             # Démarrage rapide
├── docker-compose.yml         # Configuration Docker
├── .env                       # Variables d'environnement
│
├── src/                       # Code source
│   ├── api/                   # API FastAPI
│   ├── models/                # Modèles ML
│   ├── nlp/                   # Analyse NLP
│   └── data_collection/       # Collecte de données
│
├── frontend/                  # Dashboard Next.js
│   └── src/app/page.tsx       # Page principale
│
├── models/                    # Modèles entraînés (.pkl)
├── data/                      # Données historiques (CSV)
└── scripts/                   # Scripts utilitaires
```

---

## 🆘 Dépannage

### API ne démarre pas
```bash
docker-compose logs api --tail=100
```

### Token JWT expiré
```bash
docker-compose exec -T api python /app/generate_jwt_token.py
```

### Sentiment NULL
```bash
docker-compose exec -T api python /app/collect_news.py
```

**📖 [Guide de dépannage complet →](README_COMPLET.md#dépannage)**

---

## 🎓 Pour les Agents IA

Ce système est conçu pour être compréhensible par d'autres agents IA:

1. **Architecture claire**: Modèle hybride Prophet + XGBoost + FinBERT
2. **Documentation exhaustive**: README_COMPLET.md avec tous les détails
3. **Scripts automatisés**: update_system.bat pour la mise à jour complète
4. **Logs structurés**: Tous les événements sont loggés
5. **Tests inclus**: Tests unitaires et d'intégration

**📖 [Guide pour agents IA →](README_COMPLET.md#pour-un-autre-agent-ia)**

---

## 📞 Support

- **Documentation**: [README_COMPLET.md](README_COMPLET.md)
- **Démarrage rapide**: [QUICK_START.md](QUICK_START.md)
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails

---

## 🙏 Remerciements

- **Prophet** - Facebook Research
- **XGBoost** - DMLC
- **FinBERT** - ProsusAI
- **FastAPI** - Sebastián Ramírez
- **Next.js** - Vercel
- **Supabase** - Supabase Inc.

---

**Dernière mise à jour**: 13 Mai 2026 | **Version**: 2.0.0 | **Auteur**: STE-SCPB / Afrexia
