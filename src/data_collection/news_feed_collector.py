"""
Collecteur multi-sources news cacao (RSS + HTML listing).

Echecs isoles par source. Filtre mots-cles + anti-bruit.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from src.data_collection.news_sources import (
    KEYWORD_PATTERN,
    NOISE_PATTERN,
    NEWSAPI_CONFIG,
    enabled_sources,
)

logger = logging.getLogger(__name__)

_KW_RE = re.compile(KEYWORD_PATTERN, re.IGNORECASE)
_NOISE_RE = re.compile(NOISE_PATTERN, re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text or "")
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&")
    cleaned = cleaned.replace("&#x27;", "'").replace("&quot;", '"')
    cleaned = cleaned.replace("&#8211;", "-").replace("&#038;", "&")
    cleaned = cleaned.replace("&#039;", "'").replace("&apos;", "'")
    cleaned = cleaned.replace("&ndash;", "-").replace("&mdash;", "-")
    return _WS_RE.sub(" ", cleaned).strip()


def _absolute_url(href: str, base_url: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
        return ""
    return urljoin(base_url.rstrip("/") + "/", href)


def passes_keyword_filter(title: str, description: str = "", force_keep: bool = False) -> bool:
    text = f"{title} {description}"
    if _NOISE_RE.search(text):
        return False
    if force_keep:
        return True
    return bool(_KW_RE.search(text))


def _fetch_html(url: str, timeout: int = 30, verify_ssl: bool = True) -> Tuple[Optional[str], Optional[str]]:
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError as exc:
        return None, f"curl_cffi missing: {exc}"

    try:
        resp = cffi_requests.get(
            url,
            impersonate="chrome131",
            timeout=timeout,
            verify=verify_ssl,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            },
        )
    except Exception as exc:
        return None, str(exc)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

    html = resp.text or ""
    if "Just a moment" in html or "cf-challenge" in html.lower():
        return None, "Cloudflare challenge"

    return html, None


def _parse_rss_items(xml_text: str, source_name: str, max_items: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # feedparser fallback
        try:
            import feedparser

            parsed = feedparser.parse(xml_text)
            for entry in parsed.entries[:max_items]:
                title = _strip_html(getattr(entry, "title", "") or "")
                link = getattr(entry, "link", "") or ""
                summary = _strip_html(
                    getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                )
                published = getattr(entry, "published", None) or getattr(entry, "updated", None)
                if title and link:
                    items.append(
                        {
                            "title": title,
                            "description": summary[:500],
                            "url": link,
                            "source": source_name,
                            "published_at": _normalize_date(published),
                        }
                    )
            return items
        except Exception as exc:
            logger.warning("RSS parse failed: %s", exc)
            return []

    # RSS 2.0
    channel = root.find("channel")
    nodes = channel.findall("item") if channel is not None else []
    if not nodes:
        # Atom
        ns = {"a": "http://www.w3.org/2005/Atom"}
        nodes = root.findall("a:entry", ns) or root.findall(
            "{http://www.w3.org/2005/Atom}entry"
        )

    for node in nodes[:max_items]:
        title = _strip_html(
            (node.findtext("title") or node.findtext("{http://www.w3.org/2005/Atom}title") or "")
        )
        link = (
            node.findtext("link")
            or ""
        )
        atom_link = node.find("{http://www.w3.org/2005/Atom}link")
        if atom_link is not None and atom_link.get("href"):
            link = atom_link.get("href") or link
        desc = _strip_html(
            node.findtext("description")
            or node.findtext("{http://www.w3.org/2005/Atom}summary")
            or ""
        )
        published = (
            node.findtext("pubDate")
            or node.findtext("{http://www.w3.org/2005/Atom}updated")
            or node.findtext("{http://www.w3.org/2005/Atom}published")
        )
        if title and link:
            items.append(
                {
                    "title": title,
                    "description": desc[:500],
                    "url": link.strip(),
                    "source": source_name,
                    "published_at": _normalize_date(published),
                }
            )
    return items


def _normalize_date(raw: Optional[str]) -> str:
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    raw = raw.strip()
    # Already ISO-ish
    try:
        if "T" in raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
    except Exception:
        pass
    # RFC2822 via email.utils
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(raw).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _parse_html_list(html: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    pattern = source.get("link_regex") or r'href=["\']([^"\']+)["\'][^>]*>([^<]{10,200})'
    base = source.get("base_url") or source["url"]
    max_items = int(source.get("max_items") or 15)
    seen = set()
    items: List[Dict[str, Any]] = []

    # 1) Pattern source (href + texte)
    for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
        href = match.group(1)
        title = _strip_html(match.group(2) if match.lastindex and match.lastindex >= 2 else "")
        url = _absolute_url(href, base)
        if not url:
            continue
        if not title or len(title) < 8:
            # Derive title from URL slug
            slug = urlparse(url).path.rstrip("/").split("/")[-1]
            title = _strip_html(slug.replace("-", " "))
            if len(title) < 8:
                continue
        if title.lower() in ("read more", "suite", "voir plus", "home", "news", "more"):
            continue
        key = urlparse(url)._replace(query="", fragment="").geturl().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "title": title[:200],
                "description": title[:500],
                "url": url,
                "source": source["name"],
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(items) >= max_items:
            break

    # 2) Fallback: href-only matching path hints for this source
    if len(items) < 3:
        path_hint = source.get("path_hint")
        if path_hint:
            for href in re.findall(r'href=["\']([^"\']+)["\']', html, re.I):
                if path_hint not in href:
                    continue
                url = _absolute_url(href, base)
                key = urlparse(url)._replace(query="", fragment="").geturl().rstrip("/").lower()
                if not url or key in seen:
                    continue
                slug = urlparse(url).path.rstrip("/").split("/")[-1]
                title = _strip_html(re.sub(r"^\d+-\d+-", "", slug).replace("-", " "))
                if len(title) < 8:
                    continue
                seen.add(key)
                items.append(
                    {
                        "title": title[:200],
                        "description": title[:500],
                        "url": url,
                        "source": source["name"],
                        "published_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if len(items) >= max_items:
                    break

    # ONCC prices page: synthesize one daily snapshot if no /updates/ links
    if source.get("id") == "oncc" and not items:
        price_bits = re.findall(
            r"(?:FCFA|XAF|USD|\$)\s*[0-9][0-9\s.,]*|[0-9]{3,6}\s*(?:FCFA|XAF)",
            html,
            re.I,
        )
        snippet = ", ".join(price_bits[:6]) if price_bits else "Prix officiels cacao Cameroun"
        items.append(
            {
                "title": f"ONCC — prix quotidiens cacao ({datetime.now().date().isoformat()})",
                "description": snippet[:400],
                "url": source["url"] + f"?d={datetime.now().date().isoformat()}",
                "source": source["name"],
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return items


def collect_from_source(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns journal entry:
      {id, name, ok, error, fetched, kept, rejected, articles}
    """
    sid = source["id"]
    name = source["name"]
    timeout = int(source.get("timeout") or 30)
    verify = bool(source.get("verify_ssl", True))
    force_keep = bool(source.get("force_keep", False))
    max_items = int(source.get("max_items") or 20)

    html_or_xml, err = _fetch_html(source["url"], timeout=timeout, verify_ssl=verify)
    if err or not html_or_xml:
        return {
            "id": sid,
            "name": name,
            "ok": False,
            "error": err or "empty",
            "fetched": 0,
            "kept": 0,
            "rejected": 0,
            "articles": [],
        }

    stype = source.get("type")
    if stype == "rss" or "xml" in (html_or_xml[:200].lower()) or "<rss" in html_or_xml[:500].lower():
        raw_items = _parse_rss_items(html_or_xml, name, max_items)
    else:
        raw_items = _parse_html_list(html_or_xml, source)

    kept: List[Dict[str, Any]] = []
    rejected = 0
    for art in raw_items:
        if passes_keyword_filter(art["title"], art.get("description", ""), force_keep=force_keep):
            kept.append(art)
        else:
            rejected += 1

    return {
        "id": sid,
        "name": name,
        "ok": True,
        "error": None,
        "fetched": len(raw_items),
        "kept": len(kept),
        "rejected": rejected,
        "articles": kept,
    }


def collect_newsapi(days_back: Optional[int] = None) -> Dict[str, Any]:
    import os

    import requests

    cfg = NEWSAPI_CONFIG
    if not cfg.get("enabled", True):
        return {
            "id": "newsapi",
            "name": "NewsAPI",
            "ok": False,
            "error": "disabled",
            "fetched": 0,
            "kept": 0,
            "rejected": 0,
            "articles": [],
        }

    key = os.getenv(cfg["api_key_env"]) or os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if not key:
        return {
            "id": "newsapi",
            "name": "NewsAPI",
            "ok": False,
            "error": "NEWSAPI_KEY missing",
            "fetched": 0,
            "kept": 0,
            "rejected": 0,
            "articles": [],
        }

    from datetime import timedelta

    days = days_back if days_back is not None else int(cfg.get("days_back") or 7)
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": cfg["query"],
        "sortBy": "publishedAt",
        "from": from_date,
        "pageSize": int(cfg.get("page_size") or 30),
        "apiKey": key,
        "domains": cfg.get("domains"),
    }
    # Drop None
    params = {k: v for k, v in params.items() if v is not None}

    try:
        resp = requests.get(
            f"{os.getenv('NEWSAPI_URL', 'https://newsapi.org/v2')}/everything",
            params=params,
            timeout=20,
        )
        data = resp.json() if resp.content else {}
    except Exception as exc:
        return {
            "id": "newsapi",
            "name": "NewsAPI",
            "ok": False,
            "error": str(exc),
            "fetched": 0,
            "kept": 0,
            "rejected": 0,
            "articles": [],
        }

    if resp.status_code != 200 or data.get("status") != "ok":
        return {
            "id": "newsapi",
            "name": "NewsAPI",
            "ok": False,
            "error": data.get("message") or f"HTTP {resp.status_code}",
            "fetched": 0,
            "kept": 0,
            "rejected": 0,
            "articles": [],
        }

    raw = data.get("articles") or []
    kept = []
    rejected = 0
    for a in raw:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        url = a.get("url") or ""
        if not title or not url or title == "[Removed]":
            rejected += 1
            continue
        if not passes_keyword_filter(title, desc):
            rejected += 1
            continue
        src = a.get("source") or {}
        kept.append(
            {
                "title": title,
                "description": desc,
                "content": a.get("content") or desc or "",
                "url": url,
                "source": src.get("name") or "NewsAPI",
                "published_at": a.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "id": "newsapi",
        "name": "NewsAPI",
        "ok": True,
        "error": None,
        "fetched": len(raw),
        "kept": len(kept),
        "rejected": rejected,
        "articles": kept,
    }


def collect_all_sources() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (articles_deduped, journal_entries).
    """
    journal: List[Dict[str, Any]] = []
    all_articles: List[Dict[str, Any]] = []

    for source in enabled_sources():
        entry = collect_from_source(source)
        journal.append(entry)
        all_articles.extend(entry["articles"])
        if entry["ok"]:
            logger.info(
                "News %s: fetched=%s kept=%s rejected=%s",
                entry["name"],
                entry["fetched"],
                entry["kept"],
                entry["rejected"],
            )
        else:
            logger.warning("News %s failed: %s", entry["name"], entry["error"])

    newsapi_entry = collect_newsapi()
    journal.append(newsapi_entry)
    all_articles.extend(newsapi_entry["articles"])

    # Dedup by URL
    seen = set()
    deduped = []
    for art in all_articles:
        key = (art.get("url") or "").split("?")[0].rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(art)

    return deduped, journal
