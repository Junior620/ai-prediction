# 📋 Résumé Final - Système de Prédiction Cocoa (MAJ 21 Mai 2026)

## ✅ Ce qui a été fait : Intégration de NeuralForecast (N-HiTS)

Le système de prédiction a été significativement modernisé pour passer d'une architecture à 2 moteurs vers un **système d'Ensemble Learning hybride à 3 moteurs**.

### 1. Le Troisième Moteur : N-HiTS
- **Technologie** : N-HiTS (Neural Hierarchical Interpolation for Time Series) de la librairie NeuralForecast.
- **Rôle** : Modèle de Deep Learning spécialisé dans les prédictions multi-horizons, particulièrement efficace pour capturer la volatilité sur le court et moyen terme.
- **Entraînement** : Création d'un script autonome (`train_nhits.py`) qui s'entraîne sur l'historique complet, gère les outliers, et sauvegarde le modèle dans un dossier horodaté.
- **Performances** : N-HiTS offre une grande précision à court terme (ex: erreur de ~$20 sur la prédiction J+1 lors des derniers tests).

### 2. Architecture "Ensemble" Hybride
**Fichier**: `src/models/improved_price_predictor.py`
- L'algorithme combine désormais les 3 moteurs avec une pondération optimisée :
  - **40% XGBoost** (Pour capturer les relations non linéaires complexes)
  - **40% N-HiTS** (Pour la dynamique deep-learning multi-horizon)
  - **20% Prophet** (Baseline robuste pour la tendance générale)
  - **+ Sentiment NLP** (Ajustement final basé sur les actualités FinBERT)
- **Fallback Gracieux** : Le système est résilient. Si N-HiTS n'est pas disponible (ou échoue au chargement), l'API bascule automatiquement sur l'ancien système (Prophet + XGBoost) sans interruption de service.

### 3. Backend API et Auto-Découverte
**Fichier**: `src/api/app.py`
- L'API FastAPI au démarrage scanne le dossier `models/` pour trouver et charger :
  1. Le dernier modèle Prophet (`prophet_improved_*.pkl`)
  2. Le dernier modèle XGBoost (`xgboost_improved_*.pkl`)
  3. Le dernier dossier modèle N-HiTS (`nhits_*/`)
- Le retour API inclut désormais un champ `components` qui détaille la contribution exacte de chaque moteur, offrant une totale transparence.

### 4. Automatisation Complète
**Fichier**: `update_system.bat`
- Le script batch d'automatisation inclut désormais l'entraînement de N-HiTS.
- **Ordre d'exécution** :
  1. Collecte des données (Prix + News)
  2. Entraînement XGBoost / Prophet (`train_hybrid_improved.py`)
  3. Entraînement N-HiTS (`train_nhits.py`) — géré en `--unbuffered` et `UTF-8` pour éviter les crashs Windows.
  4. Redémarrage du conteneur API Docker.

### 5. Frontend & UI (Next.js)
**Fichier**: `frontend/src/app/page.tsx` et `frontend/src/types/api.ts`
- Le Dashboard Next.js détecte dynamiquement le nombre de moteurs actifs.
- L'interface affiche **"3 moteurs"** (si N-HiTS est présent) ou **"2 moteurs"** de façon dynamique.
- Le type TypeScript a été mis à jour pour supporter le nouveau payload de l'API.

### 6. Conteneurisation (Docker)
**Fichier**: `Dockerfile`
- Ajout de la librairie `neuralforecast>=3.1.0` (et `torch`) pour permettre à l'image Docker de charger les modèles N-HiTS et d'effectuer les prédictions en environnement isolé.
- L'image a été reconstruite avec succès.

---

## 🎯 État actuel du système (21 Mai 2026)

### Services opérationnels

#### API FastAPI (Docker)
- **URL**: http://localhost:8000
- **Moteurs**: 3 actifs (Prophet, XGBoost, N-HiTS)
- **Sentiment**: ✅ NLP FinBERT opérationnel
- **Authentification**: ✅ JWT fonctionnelle (tokens 7 jours)

#### Base de données Supabase
- **Prix**: Historique 2020-2026
- **News**: Connecté à NewsAPI
- **Prédictions**: Logguées pour audit

#### Dashboard Next.js
- **URL**: http://localhost:3001
- **Design**: Premium avec indicateurs dynamiques de moteurs actifs.

---

## 🚀 Comment utiliser le système au quotidien

### 1. Lancement du système (Si éteint)
```bash
# Lancer les conteneurs (API, Redis, MLFlow)
docker-compose up -d

# Lancer le Frontend (dans un autre terminal)
cd frontend
npm run dev -- -p 3001
```

### 2. Mise à jour automatique (Données + Modèles)
Il te suffit de double-cliquer sur le fichier **`update_system.bat`**. 
Il s'occupera de tout :
- Scraping des nouveaux prix et articles.
- Ré-entraînement des 3 moteurs d'intelligence artificielle.
- Redémarrage de l'API pour charger les nouveaux cerveaux.

### 3. Tester l'API manuellement
```powershell
$headers = @{
    "Authorization" = "Bearer TON_TOKEN_JWT"
    "Content-Type" = "application/json"
}

$body = @{
    market = "ICE_NY"
    horizons = @(1, 7, 30)
    include_sentiment = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Headers $headers -Body $body
```

---

## 🔧 Dépannage (Troubleshooting)

### N-HiTS ne s'affiche pas sur le Dashboard ("2 moteurs" au lieu de "3")
- **Cause 1** : L'entraînement a échoué. Vérifie le log dans la console de `update_system.bat`.
- **Cause 2** : L'API n'a pas été redémarrée. Exécute `docker restart cocoa-api`.
- **Cause 3** : Le Dockerfile n'est pas à jour. Assure-toi d'avoir exécuté `docker-compose build api` pour installer `neuralforecast` dans le conteneur.

### L'API plante au démarrage
- Affiche les logs Docker pour voir ce qui bloque :
  ```bash
  docker logs cocoa-api --tail=100
  ```
- Si c'est un problème de téléchargement de modèle HuggingFace (FinBERT), relance simplement le conteneur car le réseau peut parfois timeout.

---

## ✅ Checklist d'Intégration Terminée

- [x] Création de `train_nhits.py` sans bugs d'encodage Windows.
- [x] Ajout de l'Ensemble 40/40/20 dans le Predictor principal.
- [x] Mise à jour du loader dans FastAPI.
- [x] Mise à jour de `update_system.bat` pour le pipeline complet.
- [x] Typage TypeScript et composant UI mis à jour sur Next.js.
- [x] `Dockerfile` enrichi avec PyTorch & NeuralForecast.
- [x] Image Docker reconstruite et API redémarrée.

**🎉 SYSTÈME COMPLET ET OPÉRATIONNEL AVEC INTELLIGENCE ARTIFICIELLE À 3 MOTEURS !**

**Dernière mise à jour**: 21 Mai 2026  
**Version**: 3.0.0 (Ensemble Learning Hybride)  
**Auteur**: STE-SCPB / Afrexia
