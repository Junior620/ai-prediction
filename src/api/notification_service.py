"""Persist and publish dashboard notifications."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.api.ws_hub import REDIS_CHANNEL


def _title_for_tv(signal_type: str, brief_signal: Optional[str]) -> str:
    labels = {
        "buy": "Signal d'achat",
        "sell": "Signal de vente",
        "support_break": "Cassure de support",
        "resistance_break": "Cassure de résistance",
        "trend_change": "Changement de tendance",
    }
    base = labels.get(signal_type, signal_type.replace("_", " ").title())
    if brief_signal:
        return f"{base} · Brief {brief_signal}"
    return base


def create_tv_notification_record(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    signal_type = snapshot.get("signal_type") or "custom"
    brief_signal = snapshot.get("brief_signal")
    summary = snapshot.get("brief_summary") or snapshot.get("message") or ""
    title = _title_for_tv(str(signal_type), brief_signal)
    body = summary[:400] if isinstance(summary, str) else ""
    return {
        "market": snapshot.get("market"),
        "source": "tradingview",
        "kind": signal_type,
        "title": title,
        "body": body,
        "payload": snapshot,
        "is_read": False,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def persist_notification(supabase_client, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if supabase_client is None:
        return None
    try:
        resp = supabase_client.table("notifications").insert(record).execute()
        if resp.data:
            row = resp.data[0]
            return {
                "id": str(row.get("id")),
                "market": row.get("market"),
                "source": row.get("source") or "tradingview",
                "kind": row.get("kind"),
                "title": row.get("title"),
                "body": row.get("body"),
                "payload": row.get("payload") or {},
                "is_read": bool(row.get("is_read")),
                "created_at": str(row.get("created_at") or record.get("created_at")),
            }
    except Exception as e:
        logger.warning(f"Notification persist failed (table missing?): {e}")
    return None


def publish_notification(redis_cache, notification: Dict[str, Any]) -> None:
    """Publish to Redis so all API workers broadcast via WebSocket."""
    if redis_cache is None or redis_cache.redis_client is None:
        return
    try:
        message = {
            "type": "notification",
            "market": notification.get("market"),
            "data": notification,
        }
        redis_cache.redis_client.publish(REDIS_CHANNEL, json.dumps(message, default=str))
    except Exception as e:
        logger.warning(f"Notification publish failed: {e}")


def list_notifications(
    supabase_client,
    market: str,
    limit: int = 30,
    unread_only: bool = False,
) -> List[Dict[str, Any]]:
    if supabase_client is None:
        return []
    try:
        q = (
            supabase_client.table("notifications")
            .select("*")
            .eq("market", market)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if unread_only:
            q = q.eq("is_read", False)
        resp = q.execute()
        out = []
        for row in resp.data or []:
            out.append(
                {
                    "id": str(row.get("id")),
                    "market": row.get("market"),
                    "source": row.get("source") or "tradingview",
                    "kind": row.get("kind"),
                    "title": row.get("title"),
                    "body": row.get("body"),
                    "payload": row.get("payload") or {},
                    "is_read": bool(row.get("is_read")),
                    "created_at": str(row.get("created_at")),
                }
            )
        return out
    except Exception as e:
        logger.warning(f"Notification list failed: {e}")
        return []


def mark_notification_read(supabase_client, notification_id: str) -> bool:
    if supabase_client is None:
        return False
    try:
        supabase_client.table("notifications").update({"is_read": True}).eq(
            "id", notification_id
        ).execute()
        return True
    except Exception as e:
        logger.warning(f"Notification mark read failed: {e}")
        return False


def mark_all_notifications_read(supabase_client, market: str) -> int:
    if supabase_client is None:
        return 0
    try:
        resp = (
            supabase_client.table("notifications")
            .update({"is_read": True})
            .eq("market", market)
            .eq("is_read", False)
            .execute()
        )
        return len(resp.data or [])
    except Exception as e:
        logger.warning(f"Notification mark-all failed: {e}")
        return 0
