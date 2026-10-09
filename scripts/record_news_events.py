"""Record event versions for articles already stored in news_articles.

Idempotent. A missing events table does not fail the nightly price update.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_collection.news_sources import source_weight
from src.nlp.event_store import append_local
from src.nlp.nlp_analyzer import NLPAnalyzer


def main() -> int:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[AVERTISSEMENT] Supabase absent")
        return 0
    from supabase import create_client

    supabase = create_client(url, key)
    try:
        response = (
            supabase.table("news_articles")
            .select("title, description, content, source, url, published_at, collected_at, sentiment_score")
            .order("collected_at", desc=True)
            .limit(80)
            .execute()
        )
        articles = response.data or []
    except Exception as exc:
        print(f"[AVERTISSEMENT] Lecture news_articles impossible: {exc}")
        return 0

    versions = []
    prior = []
    for article in articles:
        if not article.get("url") or not article.get("title"):
            continue
        seen = article.get("collected_at") or datetime.now(timezone.utc).isoformat()
        version = NLPAnalyzer.classify_event(
            article.get("title") or "",
            article.get("description") or article.get("content") or "",
            source=str(article.get("source") or ""),
            source_weight=source_weight(str(article.get("source") or "")),
            sentiment_score=article.get("sentiment_score"),
            published_at=article.get("published_at"),
            published_at_verified=False,
            first_seen_at=seen,
            as_of=seen,
            prior_visible=prior,
        )
        version["url"] = article["url"]
        versions.append(version)
        prior.append(version)
    added = append_local(versions)
    print(f"[OK] {added} versions ajoutees, {len(versions)} articles lus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
