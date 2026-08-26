"""
Entrainement des modeles de courbe a terme cacao (XGBoost par contrat ICE).

Usage:
  python train_futures_curve.py
  python train_futures_curve.py --symbols CCZ26.NYB,CCH27.NYB
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
    # union preserve order
    seen = set()
    out = []
    for s in from_db + from_cfg:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train futures curve XGBoost models")
    parser.add_argument(
        "--symbols",
        default="",
        help="Liste CSV de symboles Yahoo (ex: CCZ26.NYB,CCH27.NYB)",
    )
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else default_symbols()
    )
    if not symbols:
        print("[ERREUR] Aucun symbole a entrainer")
        return 1

    print("=" * 80)
    print("ENTRAINEMENT COURBE A TERME CACAO (XGBoost)")
    print("=" * 80)
    print(f"Symboles ({len(symbols)}): {', '.join(symbols)}")
    print()

    predictor = FuturesCurvePredictor()
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


if __name__ == "__main__":
    raise SystemExit(main())
