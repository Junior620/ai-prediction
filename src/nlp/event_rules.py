"""Versioned cocoa event rules. A later rules file does not rewrite an old label."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "config" / "nlp_events.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def load_rules() -> Dict[str, Any]:
    with open(RULES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def content_hash(title: str, excerpt: str) -> str:
    raw = f"{title.strip().lower()}\n{excerpt.strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hits(text: str, phrases: Sequence[str]) -> List[str]:
    found = []
    for phrase in phrases:
        if phrase and phrase.lower() in text:
            found.append(phrase.lower())
    return found


def _country(text: str, rules: Dict[str, Any]) -> Optional[str]:
    for code, names in (rules.get("countries") or {}).items():
        if _hits(text, names):
            return code
    return None


def _zone(country: Optional[str]) -> Optional[str]:
    if country in {"CI", "GH", "CM", "NG"}:
        return "west_africa"
    if country:
        return "other"
    return None


def _category(text: str, rules: Dict[str, Any]) -> tuple[str, List[str]]:
    best = "other"
    best_hits: List[str] = []
    for name, phrases in (rules.get("categories") or {}).items():
        hits = _hits(text, phrases)
        if len(hits) > len(best_hits):
            best = name
            best_hits = hits
    return best, best_hits


def economic_direction(text: str, rules: Dict[str, Any]) -> str:
    """Price direction from wording. A category never forces up or down."""
    up = _hits(text, rules.get("price_up_cues") or [])
    down = _hits(text, rules.get("price_down_cues") or [])
    if up and not down:
        return "hausse"
    if down and not up:
        return "baisse"
    return "incertain"


def classify_article(
    title: str,
    excerpt: str = "",
    *,
    source: str = "",
    source_weight: float = 1.0,
    sentiment_score: Optional[float] = None,
    published_at: Any = None,
    published_at_verified: bool = False,
    observed_at: Any = None,
    observed_source: Optional[str] = None,
    first_seen_at: Any = None,
    as_of: Any = None,
    prior_visible: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Classify one article. Surprise stays empty until enough past events exist."""
    rules = load_rules()
    text = f"{title} {excerpt}".lower()
    category, hits = _category(text, rules)
    country = _country(text, rules)
    direction = economic_direction(text, rules)
    reliability = float(source_weight)
    severity = round(min(5.0, (len(hits) or (1 if category != "other" else 0)) * reliability), 3)
    moment = parse_utc(as_of) or utc_now()
    published = parse_utc(published_at) if published_at_verified else None
    observed = parse_utc(observed_at)
    first_seen = parse_utc(first_seen_at)
    if first_seen is not None:
        available_at, available_source = first_seen, "first_seen"
    elif observed is not None:
        available_at, available_source = observed, observed_source or "observed"
    else:
        available_at, available_source = None, None

    prior = list(prior_visible or [])
    cluster_days = int(rules.get("cluster_days") or 5)
    event_key = _match_event_key(
        title, category, country, available_at or moment, prior, cluster_days
    )
    novelty = 1.0
    if event_key and any(row.get("event_key") == event_key for row in prior):
        novelty = 0.0
    surprise = _surprise(category, prior, moment, int(rules.get("min_history_for_surprise") or 8))
    duration = None
    if category != "other":
        duration = int((rules.get("duration_days") or {}).get(category) or 0) or None
    digest = content_hash(title, excerpt)
    return {
        "rules_version": rules["rules_version"],
        "title": title.strip(),
        "excerpt": (excerpt or "")[:500],
        "source": source,
        "source_weight": reliability,
        "category": category,
        "country": country,
        "zone": _zone(country),
        "severity": severity,
        "reliability": reliability,
        "novelty": novelty,
        "duration_days": duration,
        "sentiment_score": None if sentiment_score is None else float(sentiment_score),
        "economic_direction": direction,
        "surprise": surprise,
        "important": bool(severity >= float(rules.get("important_severity") or 1.5) and category != "other"),
        "published_at": published.isoformat() if published else None,
        "published_at_verified": bool(published_at_verified and published is not None),
        "observed_at": observed.isoformat() if observed else None,
        "observed_source": observed_source if observed is not None else None,
        "first_seen_at": first_seen.isoformat() if first_seen else None,
        "available_at": available_at.isoformat() if available_at else None,
        "available_source": available_source,
        "content_hash": digest,
        "event_key": event_key,
        "classified_at": moment.isoformat(),
    }


def _tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{4,}", title.lower())
    stop = {"cocoa", "cacao", "with", "from", "that", "this", "after", "says"}
    return {word for word in words if word not in stop}


def _match_event_key(
    title: str,
    category: str,
    country: Optional[str],
    when: datetime,
    prior: Sequence[Dict[str, Any]],
    cluster_days: int,
) -> str:
    tokens = _tokens(title)
    for row in prior:
        if row.get("category") != category or row.get("country") != country:
            continue
        seen = parse_utc(row.get("available_at") or row.get("classified_at"))
        if seen is None or abs((when - seen).total_seconds()) > cluster_days * 86400:
            continue
        other = _tokens(str(row.get("title") or ""))
        if not tokens or not other:
            continue
        overlap = len(tokens & other) / max(1, min(len(tokens), len(other)))
        if overlap >= 0.6 and row.get("event_key"):
            return str(row["event_key"])
    base = "|".join(sorted(tokens)) or title.lower()
    raw = f"{category}|{country or '-'}|{base}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _surprise(
    category: str,
    prior: Sequence[Dict[str, Any]],
    moment: datetime,
    minimum: int,
) -> Optional[str]:
    same = [
        row for row in prior
        if row.get("category") == category and parse_utc(row.get("available_at")) is not None
        and parse_utc(row.get("available_at")) <= moment
    ]
    if len(same) < minimum:
        return None
    recent = [
        row for row in same
        if (moment - parse_utc(row["available_at"])).total_seconds() <= 30 * 86400
    ]
    if not recent:
        return "unexpected"
    return "expected"
