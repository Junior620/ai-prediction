"""A cold revision score must leave the other routes responsive."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.revision_jobs import revision_value_off_loop


def _app(delay: float, calls: list[str]):
    app = FastAPI()

    def compute(market_id: str) -> dict:
        calls.append(market_id)
        time.sleep(delay)
        return {"market": market_id, "as_of": None, "horizons": []}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/v1/forecast-tracking")
    async def tracking():
        return {"market": "ICE_NY", "found": False}

    @app.get("/api/v1/revision-value")
    async def revision():
        return await revision_value_off_loop("cocoa", compute)

    return app


@pytest.mark.asyncio
async def test_a_cold_revision_does_not_block_health_or_forecasts():
    calls: list[str] = []
    app = _app(0.6, calls)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        started = time.perf_counter()
        revision = asyncio.create_task(client.get("/api/v1/revision-value"))
        await asyncio.sleep(0.05)
        health, tracking = await asyncio.gather(
            client.get("/health"),
            client.get("/api/v1/forecast-tracking"),
        )
        answered = time.perf_counter() - started
        payload = await revision
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert tracking.status_code == 200
    assert tracking.json()["found"] is False
    assert payload.status_code == 200
    assert answered < 0.4


@pytest.mark.asyncio
async def test_concurrent_revision_requests_share_one_calculation():
    calls: list[str] = []
    app = _app(0.4, calls)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.get("/api/v1/revision-value"),
            client.get("/api/v1/revision-value"),
        )
    assert first.json() == second.json()
    assert calls == ["cocoa"]


@pytest.mark.asyncio
async def test_cancelling_one_reader_still_delivers_the_result_to_the_others():
    from src.api.revision_jobs import _IN_FLIGHT, revision_value_off_loop

    _IN_FLIGHT.clear()

    def compute(market_id: str) -> dict:
        time.sleep(0.25)
        return {"market": market_id, "ok": True}

    owner = asyncio.create_task(revision_value_off_loop("cocoa", compute))
    await asyncio.sleep(0.05)
    reader = asyncio.create_task(revision_value_off_loop("cocoa", compute))
    await asyncio.sleep(0.05)
    owner.cancel()
    assert await reader == {"market": "cocoa", "ok": True}
    with pytest.raises(asyncio.CancelledError):
        await owner

    waiter = asyncio.create_task(revision_value_off_loop("robusta", compute))
    await asyncio.sleep(0.05)
    secondary = asyncio.create_task(revision_value_off_loop("robusta", compute))
    await asyncio.sleep(0.05)
    secondary.cancel()
    assert await waiter == {"market": "robusta", "ok": True}
    with pytest.raises(asyncio.CancelledError):
        await secondary
    assert await revision_value_off_loop("robusta", compute) == {"market": "robusta", "ok": True}
    _IN_FLIGHT.clear()
