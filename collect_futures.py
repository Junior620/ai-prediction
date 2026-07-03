"""
Collecter les prix des contrats à terme (futures) du cacao via Yahoo Finance.
Front-month + les prochains contrats disponibles sur ICE New York.
"""

import yfinance as yf
import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Cocoa futures contract symbols (ICE New York)
# Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
# Cocoa delivery months: Mar(H), May(K), Jul(N), Sep(U), Dec(Z)
FUTURES_CONTRACTS = [
    {'symbol': 'CCN26.NYB', 'label': 'Jul 2026'},
    {'symbol': 'CCU26.NYB', 'label': 'Sep 2026'},
    {'symbol': 'CCZ26.NYB', 'label': 'Dec 2026'},
    {'symbol': 'CCH27.NYB', 'label': 'Mar 2027'},
    {'symbol': 'CCK27.NYB', 'label': 'May 2027'},
    {'symbol': 'CCN27.NYB', 'label': 'Jul 2027'},
    {'symbol': 'CCU27.NYB', 'label': 'Sep 2027'},
    {'symbol': 'CCZ27.NYB', 'label': 'Dec 2027'},
]


def fetch_futures():
    """Fetch cocoa futures prices from Yahoo Finance.
    Returns list of dicts: {'contract': str, 'symbol': str, 'price_usd': float}
    """
    contracts = []
    for c in FUTURES_CONTRACTS:
        try:
            t = yf.Ticker(c['symbol'])
            price = t.info.get('regularMarketPrice') or t.info.get('previousClose')
            if price:
                contracts.append({
                    'contract': c['label'],
                    'symbol': c['symbol'],
                    'price_usd': float(price)
                })
                print(f"   {c['label']}: ${price:,.2f}")
            else:
                print(f"   {c['label']}: no price available")
        except Exception as e:
            print(f"   {c['label']}: error - {e}")
    return contracts


def store_futures():
    """Fetch futures and store in Supabase cocoa_futures table."""
    print("   Fetching cocoa futures from Yahoo Finance...")
    data = fetch_futures()
    if not data:
        print("   No futures data fetched")
        return

    payload = {
        'data': data,
        'source': 'yahoo_finance',
        'collected_at': datetime.utcnow().isoformat()
    }
    try:
        supabase.table('cocoa_futures').insert(payload).execute()
        print(f"   Inserted {len(data)} futures contracts into Supabase")
    except Exception as e:
        print(f"   Error inserting futures: {e}")


if __name__ == '__main__':
    print("=" * 80)
    print("COLLECTE DES CONTRATS A TERME (FUTURES) - CACAO ICE")
    print("=" * 80)
    store_futures()
    print("=" * 80)
