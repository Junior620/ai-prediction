"""Backfill quotidien NASA POWER vers data/ et, si la table existe, Supabase.

La table cocoa_weather_daily doit être créée avec
sql/20261008_cocoa_weather_daily.sql. Ce script ne modifie pas le manifeste.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.data_collection.cocoa_weather import (
    COCOA_WEATHER_POINTS,
    blend_country_daily,
    fetch_nasa_power_point,
    rows_for_supabase,
)

LOCAL_POINTS = ROOT / "data" / "cocoa_weather_daily.csv"
LOCAL_BLEND = ROOT / "data" / "cocoa_weather_blend.csv"


def _upsert(supabase, rows: list) -> int:
    inserted = 0
    for start in range(0, len(rows), 500):
        chunk = rows[start : start + 500]
        supabase.table("cocoa_weather_daily").upsert(chunk, on_conflict="date,region").execute()
        inserted += len(chunk)
        print(f"  ... {inserted}/{len(rows)} lignes")
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Historique météo cacao NASA POWER")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()

    frames = []
    for point in COCOA_WEATHER_POINTS:
        print(f"[INFO] {point['region']} ({point['country']})")
        try:
            frames.append(fetch_nasa_power_point(point, args.start, args.end))
        except Exception as exc:
            print(f"[ERREUR] {point['region']}: {exc}")
            return 1
    points = pd.concat(frames, ignore_index=True)
    points = points.dropna(subset=["precip_mm", "temp_c"])
    LOCAL_POINTS.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(LOCAL_POINTS, index=False)
    blend = blend_country_daily(points)
    blend.to_csv(LOCAL_BLEND, index=False)
    print(f"[OK] {len(points)} lignes point -> {LOCAL_POINTS}")
    print(f"[OK] {len(blend)} jours pondérés -> {LOCAL_BLEND}")

    import os

    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[AVERTISSEMENT] Supabase absent : CSV local seulement")
        return 0
    supabase = create_client(url, key)
    try:
        supabase.table("cocoa_weather_daily").select("date").limit(1).execute()
    except Exception as exc:
        print(f"[AVERTISSEMENT] Table cocoa_weather_daily absente ({exc})")
        print("         Appliquer sql/20261008_cocoa_weather_daily.sql")
        return 0
    written = _upsert(supabase, rows_for_supabase(points))
    print(f"[OK] {written} lignes upsertées")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
