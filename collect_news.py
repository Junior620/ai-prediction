"""
Collecte multi-sources des news cacao (veille SCPB) + sentiment + Supabase.

Sources MVP: ONCC, ICCO, CCC CI, COCOBOD, Ecofin, ConfectioneryNews,
Investir au Cameroun (RSS) + NewsAPI filtre.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

# Evite UnicodeEncodeError sur consoles Windows cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.data_collection.news_feed_collector import collect_all_sources
from src.data_collection.news_sources import source_weight
from src.data_collection.sentiment_scoring import score_sentiment
from src.nlp.event_store import append_local
from src.nlp.nlp_analyzer import NLPAnalyzer


# Sources francophones connues (traduction forcee meme si langdetect hesite)
_FR_SOURCES = {
    "oncc",
    "conseil_cafe_cacao",
    "ecofin",
    "investir_cameroun",
}


def _write_journal(
    journal: list,
    saved: int,
    analyzed: int,
    *,
    translated: int = 0,
) -> Path:
    logs = Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    path = logs / f"news_collection_{day}.json"
    payload = {
        "collected_at": datetime.now().isoformat(),
        "sources_ok": [j["name"] for j in journal if j.get("ok")],
        "sources_failed": [
            {"name": j["name"], "error": j.get("error")} for j in journal if not j.get("ok")
        ],
        "sources_with_novelty": [j["name"] for j in journal if j.get("kept", 0) > 0],
        "sources_detail": [
            {
                "id": j["id"],
                "name": j["name"],
                "ok": j["ok"],
                "error": j.get("error"),
                "fetched": j.get("fetched", 0),
                "kept": j.get("kept", 0),
                "rejected_by_filter": j.get("rejected", 0),
            }
            for j in journal
        ],
        "articles_analyzed": analyzed,
        "articles_translated_fr_en": translated,
        "articles_saved": saved,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _record_events(articles: list) -> None:
    """Store event versions. A failure here does not fail the news collection."""
    if not articles:
        return
    try:
        seen_at = datetime.now(timezone.utc).isoformat()
        versions = []
        prior: list = []
        for article in articles:
            payload = dict(article)
            payload["first_seen_at"] = payload.get("first_seen_at") or seen_at
            payload["source_weight"] = source_weight(str(payload.get("source") or ""))
            version = NLPAnalyzer.classify_event(
                payload.get("title") or "",
                payload.get("description") or "",
                source=str(payload.get("source") or ""),
                source_weight=float(payload["source_weight"]),
                sentiment_score=payload.get("sentiment_score"),
                published_at=payload.get("published_at"),
                published_at_verified=bool(payload.get("published_at_verified")),
                first_seen_at=payload["first_seen_at"],
                as_of=payload["first_seen_at"],
                prior_visible=prior,
            )
            version["url"] = str(payload.get("url") or "")
            if version["url"]:
                versions.append(version)
                prior.append(version)
        added = append_local(versions)
        _try_supabase(versions)
        print(f"[OK] Evenements NLP: {added} nouvelles versions locales")
    except Exception as exc:
        print(f"[AVERTISSEMENT] Classement evenementiel ignore: {exc}")


def _try_supabase(versions: list) -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key or not versions:
        return
    try:
        supabase = create_client(url, key)
        supabase.table("cocoa_news_event_versions").select("url").limit(1).execute()
    except Exception as exc:
        print(f"[AVERTISSEMENT] cocoa_news_event_versions absente ({exc})")
        return
    for row in versions:
        try:
            supabase.table("cocoa_news_events").upsert(
                {"event_key": row["event_key"], "url": row["url"], "source": row["source"]},
                on_conflict="url",
            ).execute()
            supabase.table("cocoa_news_event_versions").upsert(_version_row(row), on_conflict="url,rules_version,content_hash").execute()
        except Exception as exc:
            print(f"[AVERTISSEMENT] version non enregistree: {exc}")


def _version_row(row: dict) -> dict:
    keep = [
        "url", "event_key", "source", "source_weight", "title", "excerpt", "category",
        "country", "zone", "severity", "reliability", "novelty", "duration_days",
        "sentiment_score", "economic_direction", "surprise", "published_at",
        "published_at_verified", "observed_at", "observed_source", "first_seen_at",
        "available_at", "available_source", "content_hash", "rules_version", "classified_at",
    ]
    return {key: row.get(key) for key in keep}


def main() -> int:
    print("=" * 80)
    print("COLLECTE NEWS CACAO - veille multi-sources SCPB")
    print("=" * 80)

    print("\n[1/3] Collecte multi-sources (RSS/HTML + NewsAPI filtre)...")
    articles, journal = collect_all_sources()

    for j in journal:
        if j.get("ok"):
            print(
                f"   [OK] {j['name']}: {j.get('fetched', 0)} bruts -> "
                f"{j.get('kept', 0)} retenus ({j.get('rejected', 0)} filtres)"
            )
        else:
            print(f"   [KO] {j['name']}: {j.get('error')}")

    print(f"\n   Total apres dedup: {len(articles)} articles")

    if not articles:
        print("[AVERTISSEMENT] Aucun article retenu - verifiez reseau / sources")
        path = _write_journal(journal, saved=0, analyzed=0, translated=0)
        print(f"Journal: {path}")
        return 0  # non bloquant pour update_system

    print("\n[2/3] Analyse du sentiment (FR->EN si besoin)...")
    analyzed = []
    translated_count = 0
    for art in articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        if not title:
            continue
        try:
            src_id = (art.get("source_id") or art.get("source") or "").lower()
            force_fr = any(k in src_id for k in _FR_SOURCES)
            score, label, was_tr = score_sentiment(
                title, desc, force_translate=force_fr
            )
            if was_tr:
                translated_count += 1
            art["sentiment_score"] = score
            art["sentiment_label"] = label
            art["sentiment_translated"] = was_tr
            analyzed.append(art)
            flag = " [FR->EN]" if was_tr else ""
            print(
                f"   {title[:60]}... -> {label} ({score:.2f}) "
                f"[{art.get('source')}]{flag}"
            )
        except Exception as exc:
            print(f"   [WARN] sentiment: {title[:40]}... ({exc})")

    print(
        f"[OK] {len(analyzed)} articles analyses "
        f"({translated_count} traduits FR->EN)"
    )

    print("\n[3/3] Sauvegarde dans Supabase...")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERREUR] SUPABASE_URL / SUPABASE_KEY manquants")
        return 1

    supabase = create_client(url, key)
    saved = 0
    for article in analyzed:
        try:
            existing = (
                supabase.table("news_articles")
                .select("id")
                .eq("url", article["url"])
                .execute()
            )
            if existing.data:
                print(f"   [SKIP] Deja existant: {article['title'][:50]}...")
                continue

            insert_data = {
                "collected_at": datetime.now().isoformat(),
                "title": article["title"],
                "description": article.get("description", "") or "",
                "content": article.get("content")
                or article.get("description", "")
                or article["title"],
                "source": article.get("source") or "unknown",
                "url": article["url"],
                "published_at": article.get("published_at") or datetime.now().isoformat(),
                "sentiment_score": float(article["sentiment_score"]),
                "sentiment_label": article["sentiment_label"],
                "keywords": ["cocoa", "cacao"],
                "is_high_risk": abs(float(article["sentiment_score"])) > 0.5,
            }
            supabase.table("news_articles").insert(insert_data).execute()
            saved += 1
            print(f"   [OK] Sauve: {article['title'][:50]}...")
        except Exception as exc:
            print(f"   [ERR] {exc}")

    print(f"\n[OK] {saved} nouveaux articles sauvegardes")

    if analyzed:
        weighted_sum = 0.0
        weight_total = 0.0
        for a in analyzed:
            w = source_weight(str(a.get("source") or ""))
            a["source_weight"] = w
            weighted_sum += float(a["sentiment_score"]) * w
            weight_total += w
        avg = weighted_sum / weight_total if weight_total else 0.0
        simple = sum(float(a["sentiment_score"]) for a in analyzed) / len(analyzed)
        print("\n" + "=" * 80)
        print("SENTIMENT GLOBAL DU MARCHE (pondéré SCPB)")
        print("=" * 80)
        print(f"\nScore moyen pondéré: {avg:.3f} (simple={simple:.3f}, n={len(analyzed)})")
        if avg > 0.2:
            print("Sentiment: POSITIF (marche optimiste)")
        elif avg < -0.2:
            print("Sentiment: NEGATIF (marche pessimiste)")
        else:
            print("Sentiment: NEUTRE (marche stable)")

    _record_events(analyzed)

    path = _write_journal(
        journal,
        saved=saved,
        analyzed=len(analyzed),
        translated=translated_count,
    )
    print("\n" + "=" * 80)
    print("JOURNAL DE COLLECTE")
    print("=" * 80)
    ok = [j["name"] for j in journal if j.get("ok")]
    ko = [j["name"] for j in journal if not j.get("ok")]
    novelty = [j["name"] for j in journal if j.get("kept", 0) > 0]
    print(f"Sources OK     : {', '.join(ok) or '-'}")
    print(f"Sources KO     : {', '.join(ko) or '-'}")
    print(f"Avec nouveautes: {', '.join(novelty) or '-'}")
    print(f"Fichier        : {path}")
    print("\n" + "=" * 80)
    print("[OK] Collecte terminee")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
