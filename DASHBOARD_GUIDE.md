# 🍫 Guide d'Utilisation du Dashboard

## Pour les Non-Techniques

Ce dashboard vous permet de visualiser les prédictions du prix du cacao de manière simple et intuitive.

---

## 🚀 Démarrage Rapide

### Étape 1: Démarrer l'API
```bash
docker-compose up -d
```

### Étape 2: Démarrer le Dashboard
**Double-cliquez sur**: `start_dashboard.bat`

Le dashboard s'ouvrira automatiquement dans votre navigateur à l'adresse:
**http://localhost:8501**

---

## 📊 Utilisation du Dashboard

### Interface Principale

Le dashboard est divisé en plusieurs sections:

#### 1. **Prix Actuel** 📊
- Affiche le prix actuel du cacao (ICE London)
- Montre le sentiment du marché (Positif/Négatif/Neutre)
- Indique la version du modèle utilisé

#### 2. **Prédictions** 🔮
- Cartes avec les prédictions pour 1, 7 et 30 jours
- Changement en dollars et en pourcentage
- Code couleur:
  - 🟢 Vert = Hausse
  - 🔴 Rouge = Baisse

#### 3. **Détails des Prédictions** 📋
- Tableau avec toutes les informations:
  - Prix prédit
  - Changement absolu et relatif
  - Intervalle de confiance (IC)

#### 4. **Visualisation** 📈
- Graphique interactif montrant:
  - Évolution prédite du prix
  - Intervalle de confiance (zone grisée)
  - Prix actuel (ligne pointillée)
- **Astuce**: Survolez le graphique pour voir les détails

#### 5. **Interprétation** 💡
- Résumé de la tendance générale:
  - 📈 HAUSSE
  - 📉 BAISSE
  - ➡️ STABLE
- Explication en français simple

#### 6. **Recommandations** 🎯
- Points positifs du modèle
- Points d'attention à surveiller

---

## ⚙️ Paramètres (Barre Latérale)

### Horizons de Prédiction
Cochez/décochez les horizons que vous voulez voir:
- ☑️ 1 jour
- ☑️ 7 jours
- ☑️ 30 jours

### Analyse de Sentiment
- ☑️ **Activé**: Inclut l'analyse des news dans les prédictions
- ☐ **Désactivé**: Prédictions basées uniquement sur les prix historiques

### Bouton Rafraîchir
Cliquez sur **🔄 Rafraîchir les Prédictions** pour obtenir les dernières prédictions.

---

## 🎨 Comprendre les Couleurs

### Métriques
- **Flèche verte ↗️**: Le prix augmente
- **Flèche rouge ↘️**: Le prix diminue

### Graphique
- **Ligne marron**: Prédictions du modèle
- **Zone grise**: Intervalle de confiance (95% de chances que le prix soit dans cette zone)
- **Ligne pointillée**: Prix actuel

### Tendance
- **Vert**: Tendance haussière (prix en hausse)
- **Rouge**: Tendance baissière (prix en baisse)
- **Bleu**: Tendance stable (prix stable)

---

## 📱 Fonctionnalités Interactives

### Graphique
- **Zoom**: Cliquez et glissez sur le graphique
- **Pan**: Maintenez Shift et glissez
- **Reset**: Double-cliquez sur le graphique
- **Détails**: Survolez les points pour voir les valeurs exactes

### Tableau
- **Tri**: Cliquez sur les en-têtes de colonnes
- **Recherche**: Utilisez la barre de recherche en haut du tableau

---

## ❓ Questions Fréquentes

### Q: Que signifie "Intervalle de Confiance" ?
**R**: C'est la plage dans laquelle le prix a 95% de chances de se trouver. Plus la plage est large, plus l'incertitude est grande.

### Q: Pourquoi les prédictions changent-elles ?
**R**: Le modèle est réentraîné régulièrement avec les nouvelles données. Les prédictions s'ajustent en fonction des tendances récentes.

### Q: Le sentiment du marché, c'est quoi ?
**R**: C'est l'analyse automatique des news sur le cacao. Un sentiment positif indique des news favorables, un sentiment négatif indique des news défavorables.

### Q: Quelle est la précision du modèle ?
**R**: Le modèle a une précision de 94.55% (MAPE 5.45%), ce qui est excellent pour un marché aussi volatil que le cacao.

### Q: Puis-je faire confiance aux prédictions ?
**R**: Les prédictions sont basées sur des données historiques et des algorithmes avancés, mais le marché du cacao est très volatil. Utilisez ces prédictions comme un outil d'aide à la décision, pas comme une certitude absolue.

---

## 🔧 Dépannage

### Le dashboard ne démarre pas
1. Vérifiez que l'API est démarrée: `docker-compose ps`
2. Vérifiez que le port 8501 n'est pas utilisé
3. Réessayez: `start_dashboard.bat`

### Erreur "Erreur de connexion"
1. Vérifiez que l'API est accessible: http://localhost:8000
2. Redémarrez l'API: `docker-compose restart api`
3. Rafraîchissez le dashboard

### Les prédictions ne s'affichent pas
1. Cliquez sur **🔄 Rafraîchir les Prédictions**
2. Vérifiez que au moins un horizon est sélectionné
3. Consultez les logs de l'API: `docker logs cocoa-api`

---

## 📞 Support

Pour toute question ou problème:
1. Consultez ce guide
2. Vérifiez les logs: `docker logs cocoa-api`
3. Contactez l'équipe technique

---

## 🎯 Conseils d'Utilisation

### Pour une Présentation
1. Démarrez le dashboard avant la réunion
2. Sélectionnez les horizons pertinents (1 et 7 jours pour du court terme)
3. Activez l'analyse de sentiment pour montrer l'impact des news
4. Utilisez le graphique pour expliquer la tendance

### Pour un Suivi Quotidien
1. Ouvrez le dashboard chaque matin
2. Cliquez sur **🔄 Rafraîchir** pour avoir les dernières prédictions
3. Notez la tendance générale (Hausse/Baisse/Stable)
4. Consultez les recommandations

### Pour une Analyse Approfondie
1. Sélectionnez tous les horizons (1, 7, 30 jours)
2. Activez l'analyse de sentiment
3. Examinez le tableau détaillé
4. Analysez l'intervalle de confiance sur le graphique

---

## 📚 Ressources Supplémentaires

- **Documentation API**: http://localhost:8000/docs
- **Guide Technique**: `IMPROVED_MODEL_SUCCESS.md`
- **Résultats Finaux**: `FINAL_RESULTS_MAY11.md`

---

**Date de création**: 11 Mai 2026
**Version**: 1.0.0
**Statut**: ✅ PRÊT À L'EMPLOI
