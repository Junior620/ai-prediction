"""Write one price snapshot so training and journaling share the same rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from supabase import create_client

from src.models.hybrid_features import load_price_data_from_supabase
from src.models.market_registry import get_market_config
from src.models.revision_command import frame_fingerprint

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    market = get_market_config(args.market)
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    frame = load_price_data_from_supabase(client, table_name=market.price_table)
    if frame.empty:
        print(f"[ERREUR] Aucune séance pour {market.price_table}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    stamp = {
        "market": args.market,
        "rows": int(len(frame)),
        "max_date": str(frame["date"].max())[:10],
        "data_fingerprint": frame_fingerprint(frame),
    }
    output.with_suffix(".json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"[OK] {output} {stamp['rows']} séances, dernière {stamp['max_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
