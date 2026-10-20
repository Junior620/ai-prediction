"""Run a revision score off the HTTP event loop, once per cold request."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

_GUARD = asyncio.Lock()
_IN_FLIGHT: dict[str, asyncio.Task] = {}


async def _owned_compute(
    market_id: str,
    compute: Callable[..., dict[str, Any]],
    holder: dict[str, asyncio.Task],
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(compute, market_id)
    finally:
        async with _GUARD:
            if _IN_FLIGHT.get(market_id) is holder.get("task"):
                _IN_FLIGHT.pop(market_id, None)


async def revision_value_off_loop(
    market_id: str,
    compute: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One cold calculation is shared. Cancelling one reader does not cancel the others."""
    if compute is None:
        from src.validation.revision_value import revision_value_for_market

        compute = revision_value_for_market
    async with _GUARD:
        task = _IN_FLIGHT.get(market_id)
        if task is None or task.done():
            holder: dict[str, asyncio.Task] = {}
            task = asyncio.create_task(_owned_compute(market_id, compute, holder))
            holder["task"] = task
            _IN_FLIGHT[market_id] = task
    try:
        return await asyncio.shield(task)
    finally:
        if task.done():
            async with _GUARD:
                if _IN_FLIGHT.get(market_id) is task:
                    _IN_FLIGHT.pop(market_id, None)
