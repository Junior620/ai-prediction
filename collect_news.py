"""
Collecte multi-sources des news cacao (veille SCPB) + sentiment + Supabase.

Sources MVP: ONCC, ICCO, CCC CI, COCOBOD, Ecofin, ConfectioneryNews,
Investir au Cameroun (RSS) + NewsAPI filtre.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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
from src.data_collection.sentiment_scoring import score_sentiment


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
        avg = sum(a["sentiment_score"] for a in analyzed) / len(analyzed)
        print("\n" + "=" * 80)
        print("SENTIMENT GLOBAL DU MARCHE")
        print("=" * 80)
        print(f"\nScore moyen: {avg:.3f}")
        if avg > 0.2:
            print("Sentiment: POSITIF (marche optimiste)")
        elif avg < -0.2:
            print("Sentiment: NEGATIF (marche pessimiste)")
        else:
            print("Sentiment: NEUTRE (marche stable)")

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
