"""One-off date inventory. Does not change the 2020 training cut."""
import os
from pathlib import Path

from dotenv import load_dotenv

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from supabase import create_client

from src.models.hybrid_features import load_price_data_from_supabase

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
TABLES = (
    ("cacao", "cocoa_london_prices"),
    ("robusta", "coffee_robusta_prices"),
)


def span(frame):
    if frame is None or len(frame) == 0:
        return "aucune ligne", "aucune ligne", 0
    dates = frame["date"]
    return str(dates.min().date()), str(dates.max().date()), int(len(frame))


lines = [
    "# Inventaire des séances — 8 octobre 2026",
    "",
    "La fenêtre d'entraînement et de comparaison de ce soir reste janvier 2020 → dernière séance.",
    "L'historique plus long est seulement compté ici. Il n'est pas utilisé pour choisir les réglages.",
    "",
]
for name, table in TABLES:
    full = load_price_data_from_supabase(client, min_date="1900-01-01", table_name=table)
    # load_price_data already applies the min date, so recount the unfiltered frame
    # by cleaning a copy is not available. Reload is filtered. Query the raw count
    # from the same loader result plus a second pass is the filtered view.
    kept = load_price_data_from_supabase(client, min_date="2020-01-01", table_name=table)
    full_min, full_max, full_n = span(full)
    cut_min, cut_max, cut_n = span(kept)
    lines.append(f"## {name} ({table})")
    lines.append("")
    lines.append(f"- Sans filtre 2020 : {full_n} séances, {full_min} → {full_max}")
    lines.append(f"- Avec filtre 2020-01-01 : {cut_n} séances, {cut_min} → {cut_max}")
    lines.append("")

out = ROOT / "reports" / "verification" / "date_inventory_20261008.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
