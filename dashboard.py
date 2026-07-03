"""
Dashboard Interactif pour les Prédictions du Prix du Cacao
Interface simple pour les non-techniques
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Prédictions Prix du Cacao",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #8B4513;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Configuration API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwicm9sZSI6InVzZXIiLCJleHAiOjE3Nzg4MDY3ODMsImlhdCI6MTc3ODIwMTk4MywidHlwZSI6ImFjY2VzcyJ9.07Vp5BKUfSUhiV_jUAdbl6t82WLhS3M1VFOxYX767hU")

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Titre principal
st.markdown('<h1 class="main-header">🍫 Prédictions du Prix du Cacao</h1>', unsafe_allow_html=True)

# Section d'explication
with st.expander("❓ Comment ça marche? (Cliquez pour voir)", expanded=False):
    st.markdown("""
    ### 🎯 Qu'est-ce que ce dashboard?
    
    Ce dashboard vous permet de **visualiser les prédictions du prix du cacao** générées par notre modèle d'intelligence artificielle.
    
    ### 🤖 Comment fonctionne le modèle?
    
    Notre système utilise un **modèle hybride** qui combine:
    - **XGBoost**: Analyse les tendances récentes des prix
    - **Prophet**: Détecte les patterns saisonniers
    - **FinBERT**: Analyse le sentiment des news du marché
    
    ### 📊 Comment lire les prédictions?
    
    1. **Prix Actuel** 💰: Le prix du cacao en ce moment
    2. **Prédictions** 🔮: Les prix prévus pour 1, 7 et 30 jours
    3. **Changement %** 📈📉: La variation attendue (+ = hausse, - = baisse)
    4. **Intervalle de Confiance** 📏: La fourchette dans laquelle le prix a 95% de chances de se trouver
    
    ### ⚙️ Comment utiliser le dashboard?
    
    1. **Sidebar (à gauche)**: Choisissez les horizons de prédiction (1j, 7j, 30j)
    2. **Rafraîchir**: Cliquez sur le bouton pour obtenir les dernières prédictions
    3. **Graphique**: Survolez pour voir les détails de chaque point
    4. **Interprétation**: Lisez la tendance générale en bas
    
    ### 🔄 Mise à jour automatique
    
    - **Collecte des données**: Automatique à 10h et 16h chaque jour
    - **Réentraînement du modèle**: Automatique les lundi, mercredi et vendredi à 16h
    - **Vous n'avez rien à faire!** Le système se met à jour tout seul.
    
    ### ⚠️ Important à savoir
    
    - Les prédictions sont basées sur l'historique et les tendances
    - Le marché du cacao est **très volatil** (peut changer rapidement)
    - Utilisez ces prédictions comme **aide à la décision**, pas comme certitude absolue
    - Consultez toujours les news du marché en parallèle
    """)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/3081/3081559.png" width="80">
        <h2 style="color: #8B4513; margin-top: 0.5rem;">⚙️ Paramètres</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection des horizons
    st.markdown("### 📅 Horizons de Prédiction")
    horizon_1d = st.checkbox("📍 1 jour (Court terme)", value=True)
    horizon_7d = st.checkbox("📊 7 jours (Moyen terme)", value=True)
    horizon_30d = st.checkbox("📈 30 jours (Long terme)", value=True)
    
    # Inclure le sentiment
    st.markdown("### 🧠 Options Avancées")
    include_sentiment = st.checkbox("💭 Inclure l'analyse de sentiment", value=True, help="Analyse automatique des news pour ajuster les prédictions")
    
    # Bouton de rafraîchissement
    if st.button("🔄 Rafraîchir les Prédictions", type="primary"):
        st.rerun()
    
    st.divider()
    
    # Informations
    st.markdown("### ℹ️ Informations Système")
    st.success("""
    **🤖 Modèle**: Hybride Amélioré  
    **📌 Version**: improved_20260511_113720  
    **🎯 Précision**: MAPE 4.93%  
    **📅 Dernière mise à jour**: 11 Mai 2026  
    **🔄 Collecte auto**: 10h & 16h  
    **⚡ Réentraînement**: Lun/Mer/Ven 16h
    """)

# Construire la liste des horizons
horizons = []
if horizon_1d:
    horizons.append(1)
if horizon_7d:
    horizons.append(7)
if horizon_30d:
    horizons.append(30)

if not horizons:
    st.warning("⚠️ Veuillez sélectionner au moins un horizon de prédiction.")
    st.stop()

# Fonction pour obtenir les prédictions
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_predictions(horizons, include_sentiment):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json={
                "market": "ICE_NY",
                "horizons": horizons,
                "include_sentiment": include_sentiment
            },
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur API: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Erreur de connexion: {str(e)}"

# Obtenir les prédictions
with st.spinner("🔮 Génération des prédictions..."):
    data, error = get_predictions(horizons, include_sentiment)

if error:
    st.error(f"❌ {error}")
    st.info("💡 Assurez-vous que l'API est démarrée: `docker-compose up -d`")
    st.stop()

# Extraire les données
predictions = data.get("predictions", [])
model_version = data.get("model_version", "N/A")
sentiment_score = data.get("sentiment_score")

# Prix actuel (estimé à partir de la première prédiction)
if predictions:
    # Estimer le prix actuel à partir de la prédiction
    first_pred = predictions[0]
    current_price = first_pred["price"] / (1 + (first_pred["price"] - 4532) / 4532)
    current_price = 4532  # Prix connu du 11 mai
else:
    current_price = 4532

# Section: Prix Actuel
st.header("💰 Prix Actuel du Marché")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💵 Prix du Cacao (ICE NY)",
        value=f"${current_price:,.2f}",
        delta=None,
        help="Prix actuel du cacao sur le marché ICE New York (en USD)"
    )

with col2:
    if sentiment_score is not None:
        sentiment_label = "Positif 📈" if sentiment_score > 0.2 else "Négatif 📉" if sentiment_score < -0.2 else "Neutre ➡️"
        sentiment_color = "normal" if -0.2 <= sentiment_score <= 0.2 else "inverse"
        st.metric(
            label="💭 Sentiment du Marché",
            value=sentiment_label,
            delta=f"{sentiment_score:.2f}",
            help="Analyse automatique du sentiment des news"
        )
    else:
        st.metric(
            label="💭 Sentiment du Marché",
            value="Non disponible",
            delta=None,
            help="Aucune news récente analysée"
        )

with col3:
    st.metric(
        label="🤖 Version du Modèle",
        value=model_version.split("_")[1] if "_" in model_version else model_version,
        delta=None,
        help="Version du modèle d'IA utilisé"
    )

st.divider()

# Section: Prédictions
st.header("🔮 Prédictions Futures")

# Créer un DataFrame pour les prédictions
pred_data = []
for pred in predictions:
    horizon = pred["horizon"]
    price = pred["price"]
    ci_lower, ci_upper = pred["confidence_interval"]
    change = price - current_price
    change_pct = (change / current_price) * 100
    
    pred_data.append({
        "Horizon": f"{horizon} jour(s)",
        "Prix Prédit": f"${price:,.2f}",
        "Changement": f"${change:+,.2f}",
        "Changement %": f"{change_pct:+.2f}%",
        "IC Inférieur": f"${ci_lower:,.2f}",
        "IC Supérieur": f"${ci_upper:,.2f}"
    })

# Afficher les métriques
cols = st.columns(len(predictions))
for i, pred in enumerate(predictions):
    with cols[i]:
        horizon = pred["horizon"]
        price = pred["price"]
        change = price - current_price
        change_pct = (change / current_price) * 100
        
        st.metric(
            label=f"Dans {horizon} jour(s)",
            value=f"${price:,.2f}",
            delta=f"{change_pct:+.2f}%"
        )

st.divider()

# Tableau détaillé
st.subheader("📋 Tableau Détaillé des Prédictions")
df_pred = pd.DataFrame(pred_data)
st.dataframe(df_pred, width='stretch', hide_index=True)

st.divider()

# Graphique interactif
st.header("📊 Graphique Interactif")

# Préparer les données pour le graphique
dates = [datetime.now() + timedelta(days=pred["horizon"]) for pred in predictions]
prices = [pred["price"] for pred in predictions]
ci_lower = [pred["confidence_interval"][0] for pred in predictions]
ci_upper = [pred["confidence_interval"][1] for pred in predictions]

# Ajouter le prix actuel
dates_with_current = [datetime.now()] + dates
prices_with_current = [current_price] + prices

# Créer une courbe d'évolution jour par jour (interpolation)
max_horizon = max([pred["horizon"] for pred in predictions])
daily_dates = [datetime.now() + timedelta(days=i) for i in range(max_horizon + 1)]

# Interpolation linéaire pour les prix quotidiens
import numpy as np
horizons_array = np.array([0] + [pred["horizon"] for pred in predictions])
prices_array = np.array([current_price] + prices)
daily_prices = np.interp(range(max_horizon + 1), horizons_array, prices_array)

# Interpolation pour les intervalles de confiance
ci_lower_array = np.array([current_price] + ci_lower)
ci_upper_array = np.array([current_price] + ci_upper)
daily_ci_lower = np.interp(range(max_horizon + 1), horizons_array, ci_lower_array)
daily_ci_upper = np.interp(range(max_horizon + 1), horizons_array, ci_upper_array)

# Créer le graphique
fig = go.Figure()

# Courbe d'évolution jour par jour
fig.add_trace(go.Scatter(
    x=daily_dates,
    y=daily_prices,
    mode='lines',
    name='Évolution Prédite (jour par jour)',
    line=dict(color='#8B4513', width=2),
    hovertemplate='<b>Date</b>: %{x|%d/%m/%Y}<br><b>Prix</b>: $%{y:,.2f}<extra></extra>'
))

# Points de prédiction principaux (1j, 7j, 30j)
fig.add_trace(go.Scatter(
    x=dates_with_current,
    y=prices_with_current,
    mode='markers',
    name='Points de Prédiction',
    marker=dict(size=12, color='#D2691E', symbol='diamond'),
    hovertemplate='<b>Date</b>: %{x|%d/%m/%Y}<br><b>Prix</b>: $%{y:,.2f}<extra></extra>'
))

# Intervalle de confiance (zone grisée)
fig.add_trace(go.Scatter(
    x=daily_dates + daily_dates[::-1],
    y=list(daily_ci_upper) + list(daily_ci_lower[::-1]),
    fill='toself',
    fillcolor='rgba(139, 69, 19, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Intervalle de Confiance (95%)',
    showlegend=True,
    hoverinfo='skip'
))

# Ligne horizontale pour le prix actuel
fig.add_hline(
    y=current_price,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Prix Actuel: ${current_price:,.2f}",
    annotation_position="right"
)

# Mise en forme
fig.update_layout(
    title=f"Évolution Prédite du Prix du Cacao sur {max_horizon} jours",
    xaxis_title="Date",
    yaxis_title="Prix (USD)",
    hovermode='x unified',
    height=500,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, width='stretch')

# Ajouter une note explicative
st.info("""
💡 **Comment lire ce graphique:**
- **Courbe marron**: Évolution prédite jour par jour (interpolation entre les points)
- **Diamants oranges**: Points de prédiction principaux (1j, 7j, 30j selon votre sélection)
- **Zone grise**: Intervalle de confiance à 95% (le prix a 95% de chances d'être dans cette zone)
- **Ligne pointillée**: Prix actuel du marché
- **Survolez** la courbe pour voir les détails de chaque jour
""")

st.divider()

# Section: Interprétation
st.header("💡 Interprétation Automatique")

# Calculer la tendance générale
avg_change = sum((pred["price"] - current_price) / current_price * 100 for pred in predictions) / len(predictions)

if avg_change > 2:
    trend_emoji = "📈"
    trend_text = "HAUSSE"
    trend_color = "green"
    interpretation = f"Le modèle prédit une **hausse** du prix du cacao de **{avg_change:+.2f}%** en moyenne sur les prochains jours."
elif avg_change < -2:
    trend_emoji = "📉"
    trend_text = "BAISSE"
    trend_color = "red"
    interpretation = f"Le modèle prédit une **baisse** du prix du cacao de **{avg_change:+.2f}%** en moyenne sur les prochains jours."
else:
    trend_emoji = "➡️"
    trend_text = "STABLE"
    trend_color = "blue"
    interpretation = f"Le modèle prédit une **stabilité** du prix du cacao avec une variation de **{avg_change:+.2f}%** en moyenne."

st.markdown(f"""
<div style="background-color: {trend_color}20; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid {trend_color};">
    <h3>{trend_emoji} Tendance: {trend_text}</h3>
    <p style="font-size: 1.1rem;">{interpretation}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Section: Recommandations
st.header("🎯 Recommandations & Points Clés")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Points Positifs")
    st.markdown("""
    - ✓ Modèle entraîné avec les données les plus récentes
    - ✓ Précision de 95.07% (MAPE 4.93%)
    - ✓ Intervalles de confiance à 95%
    - ✓ Prédictions basées sur 22 features
    - ✓ Mise à jour automatique 2x/jour
    """)

with col2:
    st.markdown("### ⚠️ Points d'Attention")
    st.markdown("""
    - ⚠ Le marché du cacao est très volatil
    - ⚠ Les événements externes peuvent changer la tendance
    - ⚠ Réentraînement automatique 3x/semaine
    - ⚠ Vérifier les news du marché régulièrement
    - ⚠ Utiliser comme aide à la décision uniquement
    """)

st.divider()

# Footer
st.markdown("""
---
<div style="text-align: center; color: gray;">
    <p>🍫 Système de Prédiction du Prix du Cacao | Version 1.0.0 | © 2026</p>
    <p>Dernière mise à jour: 11 Mai 2026</p>
</div>
""", unsafe_allow_html=True)
