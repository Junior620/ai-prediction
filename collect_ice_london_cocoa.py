"""
Collecte quotidienne du prix cacao ICE London -> Supabase (cocoa_london_prices).

Priorite :
1. Databento ohlcv-1d (C.v.0..3) + Open Interest + contrats
2. Playwright ICE / Investing (jour recent seulement, sans ecraser OHLCV Databento)
3. Rattrapage OI J-3..J-1 via Databento
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_collection.databento_london_collector import (
    backfill_recent_oi,
    bars_to_contract_rows,
    bars_to_supabase_rows,
    fetch_latest_curve,
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


OHLCV_KEYS = ("open", "high", "low", "volume", "open_interest")


def _result_from_databento_bar(bar) -> IceLondonResult:
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
    )


def _table_has_oi(supabase, table: str) -> bool:
    try:
        supabase.table(table).select("open_interest").limit(1).execute()
        return True
    except Exception:
        return False


def _get_row(supabase, table: str, date: str) -> Optional[Dict[str, Any]]:
    cols = "id, price, source, open, high, low, volume, open_interest"
    try:
        resp = supabase.table(table).select(cols).eq("date", date).limit(1).execute()
    except Exception:
        resp = (
            supabase.table(table)
            .select("id, price, source, open, high, low, volume")
            .eq("date", date)
            .limit(1)
            .execute()
        )
    return resp.data[0] if resp.data else None


def _merge_upsert_price(
    supabase,
    table: str,
    result: IceLondonResult,
    open_interest: Optional[float] = None,
    include_oi: bool = True,
) -> str:
    """
    Upsert intelligent :
    - Databento ecrase/complete OHLCV+OI
    - scrape ICE ne detruit jamais open/high/low/volume/open_interest Databento
    """
    existing = _get_row(supabase, table, result.date)
    row: Dict[str, Any] = {
        "date": result.date,
        "price": float(result.price),
        "symbol": result.symbol,
        "source": result.source,
        "collected_at": datetime.now().isoformat(),
    }
    for key in ("open", "high", "low", "volume"):
        val = getattr(result, key, None)
        if val is not None:
            row[key] = val
    if include_oi and open_interest is not None:
        row["open_interest"] = float(open_interest)

    if existing:
        old_price = float(existing["price"])
        old_source = (existing.get("source") or "").lower()

        if result.source != "databento" and old_source == "databento":
            # Scrape ne remplace pas une barre Databento du meme jour
            return (
                f"[SKIP] {result.date} deja Databento "
                f"({old_price:,.2f}) — scrape ignore pour OHLCV"
            )

        if result.source != "databento":
            # Conserver microstructure deja peuplee
            for key in OHLCV_KEYS:
                if existing.get(key) is not None:
                    row.pop(key, None)

        if abs(old_price - float(result.price)) < 0.01 and result.source != "databento":
            if not any(k in row for k in OHLCV_KEYS):
                return f"[SKIP] Date {result.date} deja en base ({old_price:,.2f})"

        update_payload = {k: v for k, v in row.items() if k != "date" and v is not None}
        supabase.table(table).update(update_payload).eq("date", result.date).execute()
        return (
            f"[OK] Mis a jour: {result.date} {old_price:,.2f} -> {result.price:,.2f} "
            f"({result.source})"
        )

    supabase.table(table).insert(row).execute()
    return f"[OK] Insere: {result.date} - {result.price:,.2f} ({result.source})"


def _upsert_databento_bars(supabase, table: str, bars, include_oi: bool) -> int:
    rows = bars_to_supabase_rows(bars, front_only=True, include_oi=include_oi)
    n = 0
    for row in rows:
        # Forcer merge via IceLondonResult-like path
        fake = IceLondonResult(
            price=float(row["price"]),
            date=row["date"],
            source="databento",
            symbol=row.get("symbol", "C.v.0"),
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            volume=row.get("volume"),
        )
        msg = _merge_upsert_price(
            supabase,
            table,
            fake,
            open_interest=row.get("open_interest"),
            include_oi=include_oi,
        )
        if msg.startswith("[OK]"):
            n += 1
            print(f"  {msg}")
    return n


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


def _alert(name: str, message: str, context: Optional[dict] = None) -> None:
    if not get_alert_system:
        return
    try:
        get_alert_system().send_data_source_failure_alert(name, message, context=context or {})
    except Exception:
        pass


def _check_oi_gap_alert(supabase, table: str) -> None:
    """Alerte si OI manquant sur >3 jours ouvrés récents Databento."""
    try:
        resp = (
            supabase.table(table)
            .select("date, open_interest, source")
            .order("date", desc=True)
            .limit(10)
            .execute()
        )
    except Exception:
        return
    missing = 0
    for row in resp.data or []:
        if (row.get("source") or "") != "databento":
            continue
        if row.get("open_interest") is None:
            missing += 1
        else:
            break
    if missing >= 3:
        _alert(
            "databento_oi_gap",
            f"Open Interest manquant sur {missing} dernieres barres Databento",
            {"missing": missing},
        )


def main() -> int:
    print("=" * 80)
    print("COLLECTE PRIX CACAO ICE LONDON (M3 / entreprise)")
    print("=" * 80)

    market = get_market_config("cocoa")
    ice_url = getattr(market, "ice_london_url", None) or (
        "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
    )
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    include_oi = _table_has_oi(supabase, market.price_table)

    print("\n[1/5] Dernier prix en base...")
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

    attempts: List[dict] = []
    latest_front: Optional[IceLondonResult] = None
    open_interest: Optional[float] = None
    db_bars = []

    # --- 1) Databento courbe C.v.0..3 ---
    print("\n[2/5] Databento courbe C.v.0..3 + OI...")
    t0 = datetime.now()
    db_result = fetch_latest_curve(price_bounds=market.price_bounds, include_oi=True)
    duration_db = int((datetime.now() - t0).total_seconds() * 1000)
    attempts.extend(db_result.attempts)

    if db_result.ok and db_result.bars:
        db_bars = db_result.bars
        front_bars = [b for b in db_bars if b.contract_rank == 0]
        latest_bar = max(front_bars, key=lambda b: b.date) if front_bars else db_result.latest
        if latest_bar:
            latest_front = _result_from_databento_bar(latest_bar)
            open_interest = latest_bar.open_interest
            print(
                f"[OK] Databento front {latest_front.price:,.2f} {market.unit} "
                f"({latest_front.date}, OI={open_interest}, bars={len(db_bars)})"
            )
            write_databento_journal(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": "ok",
                    "duration_ms": duration_db,
                    "price": latest_front.price,
                    "date": latest_front.date,
                    "open_interest": open_interest,
                    "n_bars": len(db_bars),
                    "attempts": db_result.attempts,
                }
            )
            print("\n[3/5] Upsert Databento (front + contrats)...")
            _upsert_databento_bars(supabase, market.price_table, db_bars, include_oi)
            _upsert_contracts(supabase, db_bars)
    else:
        print(f"[WARN] Databento indisponible ({db_result.error})")

    # --- 2) Scrape jour plus recent si besoin ---
    last_db_date = latest_front.date if latest_front else None
    today = datetime.utcnow().strftime("%Y-%m-%d")
    need_scrape = latest_front is None or (last_db_date and last_db_date < today)

    scrape_result: Optional[IceLondonResult] = None
    if need_scrape:
        print(f"\n[4/5] Scraping ICE London (jour recent)...")
        ice = fetch_ice_london_spot(url=ice_url, price_bounds=market.price_bounds)
        if ice is not None:
            scrape_result = ice
            attempts.extend(ice.attempts or [])
            print(
                f"[OK] ICE scrape {ice.price:,.2f} {market.unit} "
                f"({ice.date}, strategie={ice.strategy})"
            )
            # N'upsert que si date >= Databento (ou Databento absent)
            if last_db_date is None or ice.date >= last_db_date:
                msg = _merge_upsert_price(
                    supabase, market.price_table, ice, open_interest=None, include_oi=False
                )
                print(msg)
            else:
                print(f"[SKIP] Scrape {ice.date} plus ancien que Databento {last_db_date}")
        else:
            attempts.append({"strategy": "ice_playwright_chain", "ok": False})
            print("[WARN] Scraping ICE echoue")
    else:
        print("\n[4/5] Scrape non necessaire (Databento a jour)")

    # --- 3) Rattrapage OI J-3..J-1 ---
    print("\n[5/5] Rattrapage OI Databento (J-3..J-1)...")
    end_bf = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_bf = (datetime.utcnow() - timedelta(days=4)).strftime("%Y-%m-%d")
    try:
        bf = backfill_recent_oi(start=start_bf, end=end_bf, price_bounds=market.price_bounds)
        attempts.extend(bf.attempts)
        if bf.ok and bf.bars:
            n = _upsert_databento_bars(supabase, market.price_table, bf.bars, include_oi)
            _upsert_contracts(supabase, bf.bars)
            print(f"[OK] Rattrapage OI: {n} jours front mis a jour")
        else:
            print(f"[INFO] Pas de rattrapage OI ({bf.error})")
    except Exception as exc:
        print(f"[WARN] Rattrapage OI echoue: {exc}")

    _check_oi_gap_alert(supabase, market.price_table)

    final = scrape_result or latest_front
    journal = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok" if final else "failed",
        "strategy": final.strategy if final else None,
        "source": final.source if final else None,
        "attempts": attempts,
        "price": final.price if final else None,
        "date": final.date if final else None,
        "open_interest": open_interest,
    }

    if final is None:
        write_collection_journal(journal)
        print("[ERREUR] Collecte ICE London echouee")
        _alert("ice_london", "Databento + Playwright ont echoue", {"url": ice_url})
        return 1

    # Ecart anormal spot scrape vs derniere Databento
    if scrape_result and latest_front and scrape_result.date == latest_front.date:
        gap = abs(scrape_result.price / latest_front.price - 1.0) * 100
        if gap > 8.0:
            _alert(
                "spot_divergence",
                f"Ecart scrape/Databento {gap:.1f}% > 8%",
                {
                    "scrape": scrape_result.price,
                    "databento": latest_front.price,
                    "date": scrape_result.date,
                },
            )

    path = write_collection_journal(journal)
    print(f"\nJournal: {path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
