"""
Entrainement des modeles de courbe a terme cacao (XGBoost).

Usage:
  python train_futures_curve.py --source london
  python train_futures_curve.py --source investing
  python train_futures_curve.py --source investing --symbols CCZ26.NYB,CCH27.NYB
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_futures import FUTURES_CONTRACTS
from src.models.futures_curve_predictor import (
    DEFAULT_LONDON_MODELS_DIR,
    DEFAULT_LONDON_NAMED_MODELS_DIR,
    FuturesCurvePredictor,
    investing_to_yahoo,
)


def symbols_from_investing_snapshot() -> list[str]:
    """Derniere courbe Investing/Yahoo en base -> symboles Yahoo trainables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return []
    try:
        from supabase import create_client

        sb = create_client(url, key)
        resp = (
            sb.table("cocoa_futures")
            .select("data")
            .order("collected_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return []
        data = resp.data[0].get("data") or []
        out = []
        for row in data:
            y = investing_to_yahoo(str(row.get("symbol") or ""))
            if y:
                out.append(y)
        return out
    except Exception as exc:
        print(f"[WARN] Snapshot Supabase indisponible: {exc}")
        return []


def default_symbols() -> list[str]:
    from_db = symbols_from_investing_snapshot()
    from_cfg = [c["symbol"] for c in FUTURES_CONTRACTS]
    seen = set()
    out = []
    for s in from_db + from_cfg:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def train_london() -> int:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERREUR] SUPABASE_URL / SUPABASE_KEY manquants")
        return 1

    from supabase import create_client

    sb = create_client(url, key)
    print("=" * 80)
    print("ENTRAINEMENT COURBE A TERME CACAO LONDRES (GBP / C.v.0..3)")
    print("=" * 80)
    print(f"Dossier modeles: {DEFAULT_LONDON_MODELS_DIR}")
    print()

    predictor = FuturesCurvePredictor(source="london")
    meta = predictor.train_london_ranks(sb, ranks=(0, 1, 2, 3))

    print()
    print(f"[OK] Modeles OK: {meta.get('n_ok')}/4")
    for r in meta.get("results", []):
        status = "OK" if r.get("ok") else "SKIP"
        print(f"  [{status}] {r.get('symbol')}")
        for h, hm in (r.get("horizons") or {}).items():
            if hm.get("ok"):
                print(f"       h={h}: MAPE={hm.get('mape')}% MAE=£{hm.get('mae')}")
            else:
                print(f"       h={h}: {hm.get('reason')}")
    print()
    print(f"Sauvegarde: {predictor.models_dir}")
    print("=" * 80)
    return 0 if meta.get("n_ok", 0) > 0 else 1


def train_investing(symbols: list[str]) -> int:
    if not symbols:
        print("[ERREUR] Aucun symbole a entrainer")
        return 1

    print("=" * 80)
    print("ENTRAINEMENT COURBE A TERME CACAO NY (XGBoost / Yahoo)")
    print("=" * 80)
    print(f"Symboles ({len(symbols)}): {', '.join(symbols)}")
    print()

    predictor = FuturesCurvePredictor(source="investing")
    meta = predictor.train_many(symbols)

    print()
    print(f"[OK] Modeles OK: {meta.get('n_ok')}/{len(symbols)}")
    for r in meta.get("results", []):
        status = "OK" if r.get("ok") else "SKIP"
        print(f"  [{status}] {r.get('symbol')}")
        for h, hm in (r.get("horizons") or {}).items():
            if hm.get("ok"):
                print(f"       h={h}: MAPE={hm.get('mape')}% MAE=${hm.get('mae')}")
            else:
                print(f"       h={h}: {hm.get('reason')}")
    print()
    print(f"Sauvegarde: {predictor.models_dir}")
    print("=" * 80)
    return 0 if meta.get("n_ok", 0) > 0 else 1


def train_london_named() -> int:
    print("=" * 80)
    print("ENTRAINEMENT COURBE A TERME LONDRES NOMMEE (DEC26 / MAR27…)")
    print("=" * 80)
    print(f"Dossier modeles: {DEFAULT_LONDON_NAMED_MODELS_DIR}")
    print("Source historique: Databento raw symbols (ohlcv-1d)")
    print()

    predictor = FuturesCurvePredictor(source="london_named")
    meta = predictor.train_london_named(count=8, lookback_days=900)

    print()
    print(f"[OK] Modeles OK: {meta.get('n_ok')}/8")
    for r in meta.get("results", []):
        status = "OK" if r.get("ok") else "SKIP"
        print(f"  [{status}] {r.get('label') or r.get('symbol')}")
        for h, hm in (r.get("horizons") or {}).items():
            if hm.get("ok"):
                print(f"       h={h}: MAPE={hm.get('mape')}% MAE=£{hm.get('mae')}")
            else:
                print(f"       h={h}: {hm.get('reason')}")
    print()
    print(f"Sauvegarde: {predictor.models_dir}")
    print("=" * 80)
    return 0 if meta.get("n_ok", 0) > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train futures curve XGBoost models")
    parser.add_argument(
        "--source",
        choices=("london", "london_named", "investing"),
        default="london",
        help="london=C.v.0..3 ; london_named=DEC26… ; investing=Yahoo NY",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Liste CSV de symboles Yahoo (source investing)",
    )
    args = parser.parse_args()

    if args.source == "london":
        return train_london()
    if args.source == "london_named":
        return train_london_named()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else default_symbols()
    )
    return train_investing(symbols)


if __name__ == "__main__":
    raise SystemExit(main())
