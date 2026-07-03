"""
Collecter les news récentes sur le cacao et les sauvegarder dans Supabase
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client
from textblob import TextBlob

load_dotenv()

print("=" * 80)
print("📰 COLLECTE DES NEWS SUR LE CACAO")
print("=" * 80)

# Configuration NewsAPI
newsapi_url = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2")
newsapi_key = os.getenv("NEWSAPI_KEY")

if not newsapi_key:
    print("❌ NEWSAPI_KEY non configurée dans .env")
    exit(1)

# Collect news
print("\n[1/3] Collecte des articles de news...")

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
        print(f"❌ Erreur NewsAPI: HTTP {response.status_code}")
        print(f"   Message: {response.text}")
        exit(1)
    
    data = response.json()
    
    if data.get("status") != "ok":
        print(f"❌ Erreur NewsAPI: {data.get('message')}")
        exit(1)
    
    articles = data.get("articles", [])
    total = data.get("totalResults", 0)
    
    print(f"✅ {len(articles)} articles collectés (total disponible: {total})")
    
except Exception as e:
    print(f"❌ Erreur lors de la collecte: {e}")
    exit(1)

if not articles:
    print("❌ Aucune news collectée")
    exit(1)

# Analyze sentiment
print("\n[2/3] Analyse du sentiment...")
articles_with_sentiment = []

for article in articles:
    if article.get("title") and article.get("description"):
        text = f"{article['title']}. {article['description']}"
        
        # Simple sentiment analysis with TextBlob
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # -1 to 1
            
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            article["sentiment_score"] = sentiment_score
            article["sentiment_label"] = sentiment_label
            articles_with_sentiment.append(article)
            
            print(f"   {article['title'][:60]}... → {sentiment_label} ({sentiment_score:.2f})")
        except Exception as e:
            print(f"   ⚠️  Erreur sentiment pour: {article['title'][:50]}...")
            continue

print(f"✅ {len(articles_with_sentiment)} articles analysés")

# Save to Supabase
print("\n[3/3] Sauvegarde dans Supabase...")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

saved_count = 0
for article in articles_with_sentiment:
    try:
        # Check if article already exists (by URL)
        existing = supabase.table("news_articles").select("id").eq("url", article["url"]).execute()
        
        if existing.data:
            print(f"   ⏭️  Déjà existant: {article['title'][:50]}...")
            continue
        
        # Insert new article (matching actual Supabase schema)
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
        print(f"   ✅ Sauvegardé: {article['title'][:50]}...")
        
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}")

print(f"\n✅ {saved_count} nouveaux articles sauvegardés dans Supabase")

# Calculate aggregate sentiment
if articles_with_sentiment:
    avg_sentiment = sum(a["sentiment_score"] for a in articles_with_sentiment) / len(articles_with_sentiment)
    
    print("\n" + "=" * 80)
    print("📊 SENTIMENT GLOBAL DU MARCHÉ")
    print("=" * 80)
    print(f"\nScore moyen: {avg_sentiment:.3f}")
    
    if avg_sentiment > 0.2:
        print("Sentiment: POSITIF 📈 (marché optimiste)")
    elif avg_sentiment < -0.2:
        print("Sentiment: NÉGATIF 📉 (marché pessimiste)")
    else:
        print("Sentiment: NEUTRE ➡️ (marché stable)")

print("\n" + "=" * 80)
print("✅ Collecte terminée!")
print("=" * 80)
