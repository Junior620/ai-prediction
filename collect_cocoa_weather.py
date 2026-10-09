"""Instantané WeatherAPI pour la Côte d'Ivoire et le Ghana.

Écrit weather_data. N'entre pas dans le prix publié.
Sortie 1 si la collecte ou l'enregistrement échoue : la mise à jour de nuit continue.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_collection.cocoa_weather import SNAPSHOT_LOCATIONS
from src.data_collection.multi_source_collector import MultiSourceCollector
from src.data_collection.supabase_saver import SupabaseSaver


def main() -> int:
    collector = MultiSourceCollector()
    payload = collector.collect_weather_data(locations=list(SNAPSHOT_LOCATIONS))
    if not payload or not payload.get("locations"):
        print("[AVERTISSEMENT] Meteo WeatherAPI indisponible")
        return 1
    saved = SupabaseSaver().save_weather_data(payload)
    if not saved:
        print("[AVERTISSEMENT] Meteo non enregistree dans weather_data")
        return 1
    print(f"[OK] {len(payload['locations'])} releves meteo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
