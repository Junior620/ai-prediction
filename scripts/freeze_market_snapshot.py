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
    parser.add_argument(
        "--until",
        default="",
        help="Dernière journée complète (AAAA-MM-JJ) : les séances plus récentes ne sont pas figées.",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    market = get_market_config(args.market)
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    frame = load_price_data_from_supabase(client, table_name=market.price_table)
    if frame.empty:
        print(f"[ERREUR] Aucune séance pour {market.price_table}")
        return 1
    frame, raw_max, dropped = cut_incomplete(frame, args.until)
    if frame.empty:
        print(f"[ERREUR] Aucune séance complète jusqu'au {args.until}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    stamp = {
        "market": args.market,
        "rows": int(len(frame)),
        "max_date": str(frame["date"].max())[:10],
        "raw_max_date": raw_max,
        "until": args.until or None,
        "dropped_incomplete": dropped,
        "data_fingerprint": frame_fingerprint(frame),
    }
    output.with_suffix(".json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"[OK] {output} {stamp['rows']} séances, dernière {stamp['max_date']}")
    if dropped:
        print(f"[AVERTISSEMENT] Séance(s) incomplète(s) non figée(s) : {', '.join(dropped)}")
    return 0


def cut_incomplete(frame, until: str):
    """Retire les séances postérieures à la dernière journée complète."""
    dates = frame["date"].astype(str).str[:10]
    raw_max = str(dates.max())
    if not until:
        return frame, raw_max, []
    keep = dates <= until[:10]
    dropped = sorted(set(dates[~keep]))
    return frame.loc[keep].reset_index(drop=True), raw_max, dropped


if __name__ == "__main__":
    raise SystemExit(main())
