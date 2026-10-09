"""One-month GDELT DOC pilot, then a single GKG and Mentions slot.

March 2024 is a West African supply-stress month. seendate is an observation
time, not proof that today's article text existed then. Rows without an
observation time are archived and excluded from replay.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = ROOT / "reports" / "news_events"
DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
WEEKS = [
    ("20240301000000", "20240307235959"),
    ("20240308000000", "20240314235959"),
    ("20240315000000", "20240321235959"),
    ("20240322000000", "20240331235959"),
]
GKG_SLOT = "20240315120000"
def _doc_week(start: str, end: str) -> List[Dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(25)
        print(f"[INFO] DOC {start[:8]} essai {attempt + 1}", flush=True)
        response = requests.get(
            DOC_URL,
            params={
                "query": "cocoa",
                "mode": "artlist",
                "maxrecords": 75,
                "startdatetime": start,
                "enddatetime": end,
                "format": "json",
            },
            timeout=25,
        )
        if response.status_code == 429:
            last_error = requests.HTTPError(f"429 {response.url}", response=response)
            continue
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("articles") or [])
    if last_error:
        raise last_error
    return []


def _parse_seendate(value: str) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) < 14:
        return None
    stamp = datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return stamp.isoformat()


def _slot_zip(kind: str, stamp: str) -> bytes | None:
    name = f"{stamp}.gkg.csv.zip" if kind == "gkg" else f"{stamp}.mentions.CSV.zip"
    url = f"http://data.gdeltproject.org/gdeltv2/{name}"
    print(f"[INFO] Téléchargement {kind} {stamp}", flush=True)
    try:
        response = requests.get(url, timeout=(20, 40), stream=True)
        if response.status_code != 200:
            print(f"[AVERTISSEMENT] {kind} HTTP {response.status_code}", flush=True)
            return None
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            size += len(chunk)
            if size > 25_000_000:
                print(f"[AVERTISSEMENT] {kind} dépasse 25 Mo, échantillon arrêté", flush=True)
                response.close()
                return None
            chunks.append(chunk)
        print(f"[OK] {kind} {size} octets", flush=True)
        return b"".join(chunks)
    except Exception as exc:
        print(f"[AVERTISSEMENT] {kind} {exc}", flush=True)
        return None


def _gkg_cocoa_count(blob: bytes) -> Dict[str, int]:
    total = 0
    cocoa = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            for raw in handle:
                total += 1
                line = raw.decode("utf-8", errors="ignore").lower()
                if "cocoa" in line or "cacao" in line:
                    cocoa += 1
                if total >= 20_000:
                    break
    return {"rows_scanned": total, "cocoa_rows": cocoa, "capped": total >= 20_000}


def _mentions_match(blob: bytes, urls: set[str]) -> Dict[str, int]:
    total = 0
    matched = 0
    needles = tuple(url for url in urls if url)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            for raw in handle:
                total += 1
                if needles and any(needle.encode("utf-8") in raw for needle in needles[:10]):
                    matched += 1
                if total >= 20_000:
                    break
    return {"rows_scanned": total, "url_matches": matched, "capped": total >= 20_000}


def main() -> int:
    print("[INFO] Pause pour le quota GDELT", flush=True)
    time.sleep(30)
    articles: List[Dict[str, Any]] = []
    errors = []
    for start, end in WEEKS:
        try:
            batch = _doc_week(start, end)
            articles.extend(batch)
            print(f"[OK] DOC {start[:8]} : {len(batch)} articles", flush=True)
        except Exception as exc:
            errors.append(f"{start[:8]}: {exc}")
            print(f"[AVERTISSEMENT] DOC {start[:8]}: {exc}", flush=True)
        time.sleep(20)

    rows = []
    for article in articles:
        raw_seen = article.get("seendate") or article.get("seenDate")
        observed = _parse_seendate(str(raw_seen or ""))
        rows.append(
            {
                "url": article.get("url"),
                "title": article.get("title"),
                "source": article.get("domain"),
                "country": article.get("sourcecountry"),
                "raw_seendate": raw_seen,
                "published_at_verified": False,
                "observed_at": observed,
                "observed_source": "gdelt_seendate" if observed else None,
                "replayable": observed is not None,
            }
        )
    replayable = [row for row in rows if row["replayable"]]
    gkg = _slot_zip("gkg", GKG_SLOT)
    mentions = _slot_zip("mentions", GKG_SLOT)
    gkg_stats = _gkg_cocoa_count(gkg) if gkg else {"rows_scanned": 0, "cocoa_rows": 0, "skipped": True}
    sample_urls = {row["url"] for row in replayable[:30] if row.get("url")}
    mention_stats = (
        _mentions_match(mentions, sample_urls)
        if mentions and sample_urls
        else {"rows_scanned": 0, "url_matches": 0, "skipped": mentions is None}
    )
    summary = {
        "pilot_month": "2024-03",
        "doc_articles": len(rows),
        "doc_with_seendate": len(replayable),
        "doc_replayable_share": (len(replayable) / len(rows)) if rows else 0.0,
        "gkg_slot": GKG_SLOT,
        "gkg": gkg_stats,
        "mentions": mention_stats,
        "errors": errors,
        "note": (
            "seendate est une heure d'observation GDELT, pas la preuve que le texte "
            "actuel existait à cette heure. Sans observed_at, la ligne est exclue du rejeu."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gdelt_pilot_202403.json").write_text(
        json.dumps({"summary": summary, "articles": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Pilote GDELT mars 2024",
        "",
        f"Articles DOC : {summary['doc_articles']}",
        f"Avec seendate : {summary['doc_with_seendate']} ({summary['doc_replayable_share']:.1%})",
        f"Créneau GKG/Mentions : {GKG_SLOT}",
        f"GKG lignes cacao / lues : {gkg_stats.get('cocoa_rows')} / {gkg_stats.get('rows_scanned')}",
        f"Mentions URL reconnues : {mention_stats.get('url_matches')} / {mention_stats.get('rows_scanned')}",
        "",
        summary["note"],
    ]
    if errors:
        lines.append("")
        lines.append("Erreurs :")
        lines.extend(f"- {err}" for err in errors)
    (OUT / "gdelt_pilot_202403.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Rapport {OUT / 'gdelt_pilot_202403.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
