"""Append-only event versions. A correction adds a row and leaves the old one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.nlp.event_rules import classify_article, load_rules, utc_now

ROOT = Path(__file__).resolve().parents[2]
LOCAL_LOG = ROOT / "data" / "cocoa_news_event_versions.jsonl"


def build_version(article: Dict[str, Any], prior: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    classified = classify_article(
        article.get("title") or "",
        article.get("description") or article.get("excerpt") or article.get("content") or "",
        source=str(article.get("source") or ""),
        source_weight=float(article.get("source_weight") or 1.0),
        sentiment_score=article.get("sentiment_score"),
        published_at=article.get("published_at"),
        published_at_verified=bool(article.get("published_at_verified")),
        observed_at=article.get("observed_at"),
        observed_source=article.get("observed_source"),
        first_seen_at=article.get("first_seen_at"),
        as_of=article.get("as_of") or article.get("first_seen_at") or utc_now(),
        prior_visible=prior,
    )
    classified["url"] = str(article.get("url") or "")
    return classified


def append_local(versions: List[Dict[str, Any]], path: Path = LOCAL_LOG) -> int:
    if not versions:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add((row.get("url"), row.get("rules_version"), row.get("content_hash")))
    added = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in versions:
            key = (row.get("url"), row.get("rules_version"), row.get("content_hash"))
            if not row.get("url") or key in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(key)
            added += 1
    return added


def load_local(path: Path = LOCAL_LOG) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rules_version() -> str:
    return str(load_rules()["rules_version"])
