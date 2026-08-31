"""
Collecte quotidienne du prix cacao ICE London -> Supabase (cocoa_london_prices).

Priorite :
1. Databento ohlcv-1d (C.v.0) + Open Interest
2. Playwright ICE
3. Investing.com UK
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from src.data_collection.databento_london_collector import (
    bars_to_contract_rows,
    fetch_latest_spot,
    write_collection_journal as write_databento_journal,
)
from src.data_collection.ice_london_collector import (
    IceLondonResult,
    fetch_ice_london_spot,
    write_collection_journal,
)
from src.models.market_registry import get_market_config

try:
    from src.monitoring.alert_system import get_alert_system
except ImportError:
    get_alert_system = None  # type: ignore


def _result_from_databento(db_result) -> Optional[IceLondonResult]:
    bar = db_result.latest
    if bar is None:
        return None
    return IceLondonResult(
        price=float(bar.price),
        date=bar.date,
        source="databento",
        symbol=bar.symbol,
        strategy="databento_ohlcv",
        open=bar.open,
        high=bar.high,
        low=bar.low,
        volume=bar.volume,
        attempts=list(db_result.attempts),
    )


def _upsert_price(
    supabase,
    table: str,
    result: IceLondonResult,
    open_interest: Optional[float] = None,
    include_oi: bool = True,
) -> str:
    check = (
        supabase.table(table)
        .select("id, price")
        .eq("date", result.date)
        .execute()
    )
    row: Dict[str, Any] = {
        "date": result.date,
        "price": float(result.price),
        "symbol": result.symbol,
        "source": result.source,
        "open": result.open,
        "high": result.high,
        "low": result.low,
        "volume": result.volume,
        "collected_at": datetime.now().isoformat(),
    }
    if include_oi and open_interest is not None:
        row["open_interest"] = float(open_interest)

    if check.data:
        old = float(check.data[0]["price"])
        if abs(old - float(result.price)) < 0.01 and result.source != "databento":
            return f"[SKIP] Date {result.date} deja en base ({old:,.2f})"
        update_payload = {k: v for k, v in row.items() if k != "date" and v is not None}
        supabase.table(table).update(update_payload).eq("date", result.date).execute()
        return f"[OK] Mis a jour: {result.date} {old:,.2f} -> {result.price:,.2f} ({result.source})"

    supabase.table(table).insert(row).execute()
    return f"[OK] Insere: {result.date} - {result.price:,.2f} ({result.source})"


def _upsert_contracts(supabase, bars) -> None:
    rows = bars_to_contract_rows(bars)
    if not rows:
        return
    try:
        supabase.table("cocoa_london_contracts").upsert(
            rows, on_conflict="date,contract_rank"
        ).execute()
        print(f"[OK] {len(rows)} contrats upsertes (cocoa_london_contracts)")
    except Exception as exc:
        print(f"[WARN] cocoa_london_contracts: {exc}")


def main() -> int:
    print("=" * 80)
    print("COLLECTE PRIX CACAO ICE LONDON")
    print("=" * 80)

    market = get_market_config("cocoa")
    ice_url = getattr(market, "ice_london_url", None) or (
        "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
    )

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print("\n[1/4] Dernier prix en base...")
    last = (
        supabase.table(market.price_table)
        .select("date, price, source")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if last.data:
        print(
            f"[OK] {last.data[0]['price']:,.2f} {market.unit} "
            f"le {last.data[0]['date']} ({last.data[0].get('source')})"
        )
    else:
        print("[WARN] Aucune donnee en base")

    attempts: list = []
    result: Optional[IceLondonResult] = None
    open_interest: Optional[float] = None
    db_bars = []
    databento_stale = False

    # --- 1) Databento ---
    print("\n[2/4] Databento (ohlcv-1d C.v.0)...")
    t0 = datetime.now()
    db_result = fetch_latest_spot(price_bounds=market.price_bounds, include_oi=True)
    duration_db = int((datetime.now() - t0).total_seconds() * 1000)
    attempts.extend(db_result.attempts)
    if db_result.ok and db_result.latest:
        result = _result_from_databento(db_result)
        open_interest = db_result.latest.open_interest
        db_bars = db_result.bars
        print(
            f"[OK] Databento {result.price:,.2f} {market.unit} "
            f"({result.date}, OI={open_interest})"
        )
        write_databento_journal(
            {
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "duration_ms": duration_db,
                "price": result.price,
                "date": result.date,
                "open_interest": open_interest,
                "attempts": db_result.attempts,
            }
        )
        # Historique Databento ~24h de retard : upsert OHLCV puis scraper si plus recent dispo
        last_db_date = last.data[0]["date"] if last.data else None
        if last_db_date and result.date < str(last_db_date)[:10]:
            print(
                f"[INFO] Databento en retard ({result.date} < {last_db_date}) "
                "— upsert OHLCV puis scrape pour le jour recent"
            )
            include_oi = True
            try:
                supabase.table(market.price_table).select("open_interest").limit(1).execute()
            except Exception:
                include_oi = False
            print(_upsert_price(supabase, market.price_table, result, open_interest, include_oi))
            if db_bars:
                _upsert_contracts(supabase, db_bars)
            databento_stale = True
            result = None
            open_interest = None
    else:
        print(f"[WARN] Databento indisponible ({db_result.error}) — fallback scrape")

    # --- 2) Playwright ICE ---
    if result is None:
        label = "scrape jour recent" if databento_stale else "fallback"
        print(f"\n[3/4] Scraping ICE London ({label}) ({ice_url})...")
        t1 = datetime.now()
        ice = fetch_ice_london_spot(url=ice_url, price_bounds=market.price_bounds)
        duration_ice = int((datetime.now() - t1).total_seconds() * 1000)
        if ice is not None:
            result = ice
            attempts.extend(ice.attempts or [])
            print(
                f"[OK] ICE scrape {result.price:,.2f} {market.unit} "
                f"({result.date}, strategie={result.strategy})"
            )
        else:
            attempts.append({"strategy": "ice_playwright_chain", "ok": False, "ms": duration_ice})
            print("[WARN] Scraping ICE echoue")
            if databento_stale:
                # Au moins l'OHLCV Databento a ete upsert
                journal = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "ok_partial",
                    "strategy": "databento_stale_only",
                    "attempts": attempts,
                }
                path = write_collection_journal(journal)
                print(f"[OK] Collecte partielle (Databento seulement). Journal: {path}")
                return 0

    journal = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok" if result else "failed",
        "strategy": result.strategy if result else None,
        "source": result.source if result else None,
        "attempts": attempts,
    }

    if result is None:
        write_collection_journal(journal)
        print("[ERREUR] Collecte ICE London echouee — aucun prix insere")
        if get_alert_system:
            try:
                get_alert_system().send_data_source_failure_alert(
                    "ice_london",
                    "Databento + Playwright + Investing ont echoue",
                    context={"url": ice_url},
                )
            except Exception:
                pass
        return 1

    lo, hi = market.price_bounds
    if not (lo <= result.price <= hi):
        journal["status"] = "failed"
        journal["error"] = f"prix_hors_bornes:{result.price}"
        write_collection_journal(journal)
        print(f"[ERREUR] Prix {result.price} hors bornes ({lo}-{hi})")
        return 1

    print("\n[4/4] Insertion Supabase...")
    include_oi = True
    try:
        supabase.table(market.price_table).select("open_interest").limit(1).execute()
    except Exception:
        include_oi = False
        print("[WARN] Colonne open_interest absente — upsert OHLCV seulement")

    msg = _upsert_price(
        supabase,
        market.price_table,
        result,
        open_interest=open_interest,
        include_oi=include_oi,
    )
    print(msg)

    if db_bars:
        _upsert_contracts(supabase, db_bars)

    journal["price"] = result.price
    journal["date"] = result.date
    journal["open_interest"] = open_interest
    path = write_collection_journal(journal)
    print(f"\nJournal: {path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
