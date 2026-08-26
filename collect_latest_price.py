"""
Collecter le prix du cacao le plus récent (11 mai 2026)
"""

import yfinance as yf
from datetime import datetime, timedelta
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
from collect_futures import store_futures
print("=" * 80)
print("📊 COLLECTE DU PRIX ACTUEL DU CACAO")
print("=" * 80)

# Connexion Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Vérifier le dernier prix dans la base
print("\n[1/3] Vérification du dernier prix dans la base...")
response = supabase.table("cocoa_prices").select("date, price").order("date", desc=True).limit(1).execute()

if response.data:
    last_date = response.data[0]['date']
    last_price = response.data[0]['price']
    print(f"✅ Dernier prix en base: ${last_price:,.2f} le {last_date}")
else:
    print("❌ Aucune donnée dans la base")
    last_date = None

# Télécharger les données récentes depuis Yahoo Finance
print("\n[2/3] Téléchargement des données récentes depuis Yahoo Finance...")

# Symbole pour le cacao (ICE Cocoa Futures)
ticker = yf.Ticker("CC=F")

# Télécharger les 7 derniers jours
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"   Période: {start_date.date()} à {end_date.date()}")

try:
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        print("❌ Aucune donnée récente disponible")
    else:
        print(f"✅ {len(df)} points téléchargés")
        
        # Afficher les données
        print("\n   Données récentes:")
        print("   " + "-" * 76)
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            price = row['Close']
            print(f"   {date_str}: ${price:,.2f}")
        
        # Prix le plus récent
        latest_date = df.index[-1]
        latest_price = df['Close'].iloc[-1]
        
        print("\n" + "=" * 80)
        print(f"📊 PRIX ACTUEL: ${latest_price:,.2f} le {latest_date.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        # Insérer les nouvelles données dans Supabase
        print("\n[3/3] Insertion des nouvelles données dans Supabase...")
        
        inserted = 0
        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            price = float(row['Close'])
            
            # Vérifier si la date existe déjà
            check = supabase.table("cocoa_prices").select("id").eq("date", date_str).execute()
            
            if not check.data:
                # Insérer
                try:
                    insert_data = {
                        "date": date_str,
                        "price": price,
                        "symbol": "CC=F",
                        "source": "yahoo_finance",
                        "collected_at": datetime.now().isoformat()
                    }
                    result = supabase.table("cocoa_prices").insert(insert_data).execute()
                    inserted += 1
                    print(f"   ✅ Inséré: {date_str} - ${price:,.2f}")
                except Exception as e:
                    print(f"   ❌ Erreur insertion {date_str}: {e.response.json() if hasattr(e, 'response') else str(e)}")
            else:
                print(f"   ⏭️  Déjà existant: {date_str}")
        
        print(f"\n✅ {inserted} nouvelles données insérées")
        
        # Comparaison avec la prédiction
        if last_date and last_price:
            print("\n" + "=" * 80)
            print("📈 COMPARAISON AVEC LA PRÉDICTION")
            print("=" * 80)
            
            # Calculer le changement réel
            real_change = latest_price - last_price
            real_change_pct = (real_change / last_price) * 100
            
            print(f"\nPrix du {last_date}: ${last_price:,.2f}")
            print(f"Prix du {latest_date.strftime('%Y-%m-%d')}: ${latest_price:,.2f}")
            print(f"Changement réel: ${real_change:+,.2f} ({real_change_pct:+.2f}%)")
            
            # Prédiction du modèle (pour 1 jour)
            predicted_price = 4063.29  # Du test précédent
            predicted_change_pct = -6.78
            
            print(f"\nPrédiction du modèle (1 jour): ${predicted_price:,.2f} ({predicted_change_pct:+.2f}%)")
            
            # Erreur
            error = abs(latest_price - predicted_price)
            error_pct = (error / latest_price) * 100
            
            print(f"\nErreur de prédiction: ${error:,.2f} ({error_pct:.2f}%)")
            
            if error_pct < 5:
                print("✅ Excellente prédiction (erreur < 5%)")
            elif error_pct < 10:
                print("✅ Bonne prédiction (erreur < 10%)")
            else:
                print("⚠️  Prédiction à améliorer (erreur > 10%)")

except Exception as e:
    print(f"❌ Erreur lors du téléchargement: {e}")

print("\n" + "=" * 80)

# Collecter les contrats à terme (Investing.com, fallback Yahoo)
print("\n📊 COLLECTE DES CONTRATS A TERME (FUTURES)")
print("=" * 80)
try:
    from collect_cocoa_futures_investing import store_cocoa_futures_investing
    n = store_cocoa_futures_investing(supabase)
    if n == 0:
        print("   Fallback Yahoo Finance...")
        store_futures()
except Exception as e:
    print(f"⚠️  Investing.com futures failed: {e}")
    try:
        print("   Fallback Yahoo Finance...")
        store_futures()
    except Exception as e2:
        print(f"⚠️  Erreur collecte futures (non bloquant): {e2}")

print("\n" + "=" * 80)
print("✅ COLLECTE TERMINÉE")
print("=" * 80)
