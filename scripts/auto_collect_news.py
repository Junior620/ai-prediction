"""
Script d'automatisation pour la collecte de news sur le cacao
Exécuté automatiquement par le planificateur Windows
"""

import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from supabase import create_client
from textblob import TextBlob

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"news_collection_{datetime.now().strftime('%Y%m%d')}.log"

def log(message):
    """Écrire dans le fichier de log et afficher"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def collect_and_save_news():
    """Collecter les news et les sauvegarder dans Supabase"""
    
    log("=" * 80)
    log("📰 DÉBUT DE LA COLLECTE AUTOMATIQUE DES NEWS")
    log("=" * 80)
    
    # Configuration NewsAPI
    newsapi_url = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2")
    newsapi_key = os.getenv("NEWSAPI_KEY")
    
    if not newsapi_key:
        log("❌ ERREUR: NEWSAPI_KEY non configurée")
        return False
    
    # Collecte des news
    log("\n[1/3] Collecte des articles de news...")
    
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"{newsapi_url}/everything"
    params = {
        "q": "cocoa OR cacao OR chocolate",
        "language": "en",
        "sortBy": "publishedAt",
        "from": from_date,
        "to": to_date,
        "pageSize": 20,
        "apiKey": newsapi_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            log(f"❌ Erreur NewsAPI: HTTP {response.status_code}")
            return False
        
        data = response.json()
        
        if data.get("status") != "ok":
            log(f"❌ Erreur NewsAPI: {data.get('message')}")
            return False
        
        articles = data.get("articles", [])
        total = data.get("totalResults", 0)
        
        log(f"✅ {len(articles)} articles collectés (total disponible: {total})")
        
    except Exception as e:
        log(f"❌ Erreur lors de la collecte: {e}")
        return False
    
    if not articles:
        log("⚠️  Aucune news collectée")
        return False
    
    # Analyse du sentiment
    log("\n[2/3] Analyse du sentiment...")
    articles_with_sentiment = []
    
    for article in articles:
        if article.get("title") and article.get("description"):
            text = f"{article['title']}. {article['description']}"
            
            try:
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                if sentiment_score > 0.1:
                    sentiment_label = "positive"
                elif sentiment_score < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                article["sentiment_score"] = sentiment_score
                article["sentiment_label"] = sentiment_label
                articles_with_sentiment.append(article)
                
            except Exception as e:
                log(f"   ⚠️  Erreur sentiment: {str(e)[:50]}")
                continue
    
    log(f"✅ {len(articles_with_sentiment)} articles analysés")
    
    # Sauvegarde dans Supabase
    log("\n[3/3] Sauvegarde dans Supabase...")
    
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
    except Exception as e:
        log(f"❌ Erreur connexion Supabase: {e}")
        return False
    
    saved_count = 0
    skipped_count = 0
    error_count = 0
    
    for article in articles_with_sentiment:
        try:
            # Vérifier si l'article existe déjà
            existing = supabase.table("news_articles").select("id").eq("url", article["url"]).execute()
            
            if existing.data:
                skipped_count += 1
                continue
            
            # Insérer le nouvel article
            insert_data = {
                "collected_at": datetime.now().isoformat(),
                "title": article["title"],
                "description": article.get("description", ""),
                "content": article.get("content", "") or article.get("description", "") or "",
                "source": article["source"]["name"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "sentiment_score": float(article["sentiment_score"]),
                "sentiment_label": article["sentiment_label"],
                "keywords": ["cocoa", "cacao"],
                "is_high_risk": abs(article["sentiment_score"]) > 0.5
            }
            
            result = supabase.table("news_articles").insert(insert_data).execute()
            saved_count += 1
            
        except Exception as e:
            error_count += 1
            log(f"   ❌ Erreur insertion: {str(e)[:100]}")
    
    log(f"\n✅ Résultats:")
    log(f"   - Sauvegardés: {saved_count}")
    log(f"   - Déjà existants: {skipped_count}")
    log(f"   - Erreurs: {error_count}")
    
    # Calcul du sentiment global
    if articles_with_sentiment:
        avg_sentiment = sum(a["sentiment_score"] for a in articles_with_sentiment) / len(articles_with_sentiment)
        
        log("\n" + "=" * 80)
        log("📊 SENTIMENT GLOBAL DU MARCHÉ")
        log("=" * 80)
        log(f"Score moyen: {avg_sentiment:.3f}")
        
        if avg_sentiment > 0.2:
            log("Sentiment: POSITIF 📈 (marché optimiste)")
        elif avg_sentiment < -0.2:
            log("Sentiment: NÉGATIF 📉 (marché pessimiste)")
        else:
            log("Sentiment: NEUTRE ➡️ (marché stable)")
    
    log("\n" + "=" * 80)
    log("✅ COLLECTE AUTOMATIQUE TERMINÉE")
    log("=" * 80)
    
    return saved_count > 0


if __name__ == "__main__":
    try:
        success = collect_and_save_news()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"❌ ERREUR CRITIQUE: {e}")
        sys.exit(1)
