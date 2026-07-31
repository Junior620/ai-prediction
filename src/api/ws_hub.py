"""WebSocket hub + Redis pub/sub for live dashboard notifications."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from loguru import logger

REDIS_CHANNEL = "scpb:notifications"


class NotificationHub:
    """In-process WebSocket fan-out, fed by Redis pub/sub."""

    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._subscriber_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, market: str) -> None:
        await websocket.accept()
        key = market.upper()
        async with self._lock:
            self._connections.setdefault(key, set()).add(websocket)
        logger.info(f"WS connected market={key} total={len(self._connections.get(key, []))}")

    async def disconnect(self, websocket: WebSocket, market: str) -> None:
        key = market.upper()
        async with self._lock:
            conns = self._connections.get(key)
            if not conns:
                return
            conns.discard(websocket)
            if not conns:
                self._connections.pop(key, None)
        logger.info(f"WS disconnected market={key}")

    async def broadcast(self, market: str, message: Dict[str, Any]) -> None:
        key = market.upper()
        payload = json.dumps(message, default=str)
        async with self._lock:
            targets = list(self._connections.get(key, set()))
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws, key)

    async def broadcast_all(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            markets = list(self._connections.keys())
        for market in markets:
            await self.broadcast(market, message)

    def start_redis_subscriber(self, redis_url_host: str, redis_port: int, password: Optional[str], db: int) -> None:
        if self._subscriber_task and not self._subscriber_task.done():
            return
        self._subscriber_task = asyncio.create_task(
            self._redis_loop(redis_url_host, redis_port, password, db)
        )

    async def stop(self) -> None:
        if self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
            self._subscriber_task = None

    async def _redis_loop(
        self,
        host: str,
        port: int,
        password: Optional[str],
        db: int,
    ) -> None:
        import redis.asyncio as aioredis

        while True:
            client = None
            try:
                client = aioredis.Redis(
                    host=host,
                    port=port,
                    password=password or None,
                    db=db,
                    decode_responses=True,
                )
                pubsub = client.pubsub()
                await pubsub.subscribe(REDIS_CHANNEL)
                logger.info(f"WS Redis subscriber listening on {REDIS_CHANNEL}")
                async for raw in pubsub.listen():
                    if raw is None or raw.get("type") != "message":
                        continue
                    data = raw.get("data")
                    try:
                        message = json.loads(data) if isinstance(data, str) else data
                    except Exception:
                        continue
                    market = str(message.get("market") or "").upper()
                    if not market:
                        continue
                    await self.broadcast(market, message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"WS Redis subscriber error: {e} — retry in 3s")
                await asyncio.sleep(3)
            finally:
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        pass


notification_hub = NotificationHub()
