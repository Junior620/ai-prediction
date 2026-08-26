"""
Registre des sources news cacao (MVP veille SCPB).

Chaque source est scrapee de facon isolee (echec non bloquant).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Mots-cles requis (titre + description) pour retenir un article
KEYWORD_PATTERN = (
    r"cacao|cocoa|cocobod|icco|eudr|grindings?|arrivages?|futures?|"
    r"harvest|producer\s*price|prix\s*(producteur|fob|bord)|feves?|"
    r"chocolate\s*(market|price|supply)|deforestation|stock"
)

# Bruit a exclure meme si "chocolate" apparait
NOISE_PATTERN = (
    r"recipe|snack|cookie|dessert|restaurant|menu|disney|netflix|"
    r"fried\s*pickle|baking\s*challenge|national\s*chocolate\s*chip"
)

MVP_SOURCES: List[Dict[str, Any]] = [
    {
        "id": "oncc",
        "name": "ONCC",
        "enabled": True,
        "type": "html_list",
        "url": "https://www.oncc.cm/prices",
        "base_url": "https://www.oncc.cm",
        "link_regex": r'href=["\']([^"\']*(?:/updates/|/resources/)[^"\']*)["\'][^>]*>([^<]{8,160})',
        "verify_ssl": True,
        "max_items": 15,
        "force_keep": False,  # filtre cacao (exclut coffee-only)
    },
    {
        "id": "icco",
        "name": "ICCO",
        "enabled": True,
        "type": "html_list",
        "url": "https://www.icco.org/news/",
        "base_url": "https://www.icco.org",
        "link_regex": r'href=["\'](https?://www\.icco\.org/[^"\']+|/[a-z0-9\-]+/?)["\'][^>]*>([^<]{12,180})',
        "verify_ssl": True,
        "max_items": 20,
        "timeout": 45,
    },
    {
        "id": "ccc_ci",
        "name": "Conseil Cafe-Cacao CI",
        "enabled": True,
        "type": "html_list",
        "url": "http://conseilcafecacao.ci/",
        "base_url": "http://conseilcafecacao.ci",
        "link_regex": r'href=["\']([^"\']*(?:index\.php\?option=com_k2|/actualit)[^"\']*)["\'][^>]*>([^<]{10,180})',
        "verify_ssl": False,
        "max_items": 15,
        "timeout": 40,
    },
    {
        "id": "cocobod",
        "name": "COCOBOD",
        "enabled": True,
        "type": "html_list",
        "url": "https://cocobod.gh/news",
        "base_url": "https://cocobod.gh",
        "link_regex": r'href=["\']((?:https?://cocobod\.gh)?/news/[a-z0-9\-]+)["\'][^>]*>([^<]{10,180})',
        "path_hint": "/news/",
        "verify_ssl": True,
        "max_items": 15,
        "force_keep": True,
    },
    {
        "id": "ecofin",
        "name": "Agence Ecofin",
        "enabled": True,
        "type": "html_list",
        "url": "https://www.agenceecofin.com/cacao",
        "base_url": "https://www.agenceecofin.com",
        "link_regex": r'href=["\']((?:https?://www\.agenceecofin\.com)?/cacao/\d+-[^"\']+)["\'][^>]*>([^<]{12,200})',
        "path_hint": "/cacao/",
        "verify_ssl": True,
        "max_items": 20,
        "force_keep": True,
    },
    {
        "id": "confectionerynews",
        "name": "ConfectioneryNews",
        "enabled": True,
        "type": "html_list",
        "url": "https://www.confectionerynews.com/",
        "base_url": "https://www.confectionerynews.com",
        "link_regex": r'href=["\']((?:https?://www\.confectionerynews\.com)?/Article/[^"\']+)["\'][^>]*>([^<]{12,200})',
        "verify_ssl": True,
        "max_items": 25,
        "timeout": 40,
    },
    {
        "id": "investir_cm",
        "name": "Investir au Cameroun",
        "enabled": True,
        "type": "rss",
        "url": "https://www.investiraucameroun.com/feed/",
        "base_url": "https://www.investiraucameroun.com",
        "verify_ssl": True,
        "max_items": 20,
    },
]

NEWSAPI_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "api_key_env": "NEWSAPI_KEY",  # collect_news historique; aussi NEWS_API_KEY
    "query": (
        "(cocoa OR cacao) AND (price OR futures OR harvest OR grindings OR "
        "EUDR OR COCOBOD OR ICCO OR deficit OR stocks)"
    ),
    "domains": (
        "reuters.com,confectionerynews.com,foodnavigator.com,"
        "agenceecofin.com,businessincameroon.com,investiraucameroun.com"
    ),
    "language": None,  # FR+EN
    "page_size": 30,
    "days_back": 7,
}


def enabled_sources() -> List[Dict[str, Any]]:
    return [s for s in MVP_SOURCES if s.get("enabled", True)]
