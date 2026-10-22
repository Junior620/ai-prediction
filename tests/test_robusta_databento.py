"""Lot B : robusta par Databento, brut versionné « point-in-time », série sans saut.

Aucun appel réseau : client Databento et base simulés, tout fichier sous ``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_collection import robusta_roll
from src.data_collection.versioned_store import VersionedBarStore

ROOT = Path(__file__).resolve().parents[1]
NOV, JAN, MAR = 8185479, 8294126, 8382509
# Mercredi 22/10/2026, 06:30 UTC : Databento a publié jusqu'au 21/10 22:05.
NOW = datetime(2026, 10, 22, 6, 30, tzinfo=timezone.utc)
AVAILABLE_END = "2026-10-21T22:05:00.000000000Z"


def _bar(day, rank, instrument, close, volume, publisher=57):
    return {
        "ts_event": pd.Timestamp(day, tz="UTC"),
        "rtype": 35,
        "publisher_id": publisher,
        "instrument_id": instrument,
        "open": close - 10.0,
        "high": close + 20.0,
        "low": close - 30.0,
        "close": close,
        "volume": volume,
        "symbol": f"RC.c.{rank}",
    }


def _sessions(start, end):
    from src.models.hybrid_features import is_closed_collection_date

    return [d for d in pd.bdate_range(start, end) if not is_closed_collection_date(d)]


def _history(start="2026-09-01", end="2026-10-21", partial_day=None):
    """NOV tenu jusqu'au 13/10, puis JAN plus échangé ; JAN cote 100 $ sous NOV."""
    rows = []
    for i, day in enumerate(_sessions(start, end)):
        nov = 3400.0 + i
        jan_volume = 900 if day < pd.Timestamp("2026-10-14") else 5000
        rows.append(_bar(day, 0, NOV, nov, 3000))
        rows.append(_bar(day, 1, JAN, nov - 100.0, jan_volume))
        rows.append(_bar(day, 2, MAR, nov - 180.0, 100))
        rows.append(_bar(day, 0, NOV, nov + 999.0, 1, publisher=84))  # XOFF : jamais retenue
    if partial_day:
        rows.append(_bar(partial_day, 1, JAN, 1.0, 1))
    return pd.DataFrame(rows).set_index("ts_event")


class FakeData:
    def __init__(self, frame):
        self.frame = frame

    def to_df(self):
        return self.frame


class FakeDatabento:
    """metadata (gratuit) et timeseries (payant) : chaque appel est noté."""

    def __init__(self, bars: pd.DataFrame, cost: float = 0.004, available_end: str = AVAILABLE_END):
        self.bars = bars
        self.cost = cost
        self.available_end = available_end
        self.get_range_calls: list[dict] = []
        self.cost_calls: list[dict] = []
        self.metadata = SimpleNamespace(get_dataset_range=self._range, get_cost=self._cost)
        self.timeseries = SimpleNamespace(get_range=self._get_range)

    def _range(self, dataset):
        return {"schema": {"ohlcv-1d": {"end": self.available_end}}}

    def _cost(self, **kwargs):
        self.cost_calls.append(kwargs)
        return self.cost

    def _get_range(self, **kwargs):
        self.get_range_calls.append(kwargs)
        frame = self.bars.reset_index()
        start = pd.Timestamp(kwargs["start"], tz="UTC")
        end = pd.Timestamp(kwargs["end"], tz="UTC")
        frame = frame[(frame["ts_event"] >= start) & (frame["ts_event"] < end)]
        frame = frame[frame["symbol"].isin(kwargs["symbols"])]
        return FakeData(frame.set_index("ts_event"))


class FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self._order_desc, self._limit, self._range = db, table, [], False, None, None
        self._insert = None

    def select(self, *_):
        return self

    def gte(self, column, value):
        self.filters.append((column, value))
        return self

    def order(self, _column, desc=False):
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def insert(self, rows):
        self._insert = list(rows)
        return self

    def execute(self):
        if self._insert is not None:
            if self.db.fail_insert:
                raise RuntimeError("insert refusé")
            self.db.inserted.extend(self._insert)
            self.db.rows.extend(self._insert)
            return SimpleNamespace(data=self._insert)
        rows = sorted(self.db.rows, key=lambda r: r["date"], reverse=self._order_desc)
        for column, value in self.filters:
            rows = [r for r in rows if str(r[column]) >= value]
        if self._range:
            rows = rows[self._range[0]:self._range[1] + 1]
        if self._limit:
            rows = rows[: self._limit]
        return SimpleNamespace(data=[dict(r) for r in rows])


class FakeSupabase:
    def __init__(self, dates):
        self.rows = [{"date": d, "price": 3000.0, "source": "investing_com"} for d in dates]
        self.inserted: list[dict] = []
        self.fail_insert = False

    def table(self, name):
        assert name == "coffee_robusta_prices"
        return FakeQuery(self, name)


@pytest.fixture
def frozen_today(monkeypatch):
    """La fraîcheur compare à la date de Londres : on la fige au 22/10/2026."""
    from src.monitoring import freshness

    monkeypatch.setattr(freshness, "london_today", lambda now=None: pd.Timestamp("2026-10-22"))


def _paths(tmp_path):
    from collect_coffee_robusta_price import Paths

    return Paths(
        raw=tmp_path / "data" / "databento_robusta_calendar_raw.csv",
        series=tmp_path / "data" / "robusta_roll_series.csv",
        quarantine=tmp_path / "data" / "quarantine" / "coffee_robusta_sessions.jsonl",
        status_dir=tmp_path / "data" / "status",
        logs_dir=tmp_path / "logs",
    )


def _seed_store(paths, until="2026-10-16"):
    """Historique déjà téléchargé (lot B, --history) jusqu'au 16/10."""
    store = VersionedBarStore(paths.raw)
    store.append(_history(end=until).reset_index(), fetched_at="2026-10-17T07:00:00Z")
    return store


# --------------------------------------------------------------------------- brut versionné


def test_the_store_keeps_every_version_and_answers_as_known_at(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    first = pd.DataFrame([_bar("2026-10-20", 0, NOV, 3500.0, 10)])
    assert store.append(first, fetched_at="2026-10-20T22:30:00Z") == 1
    assert store.append(first, fetched_at="2026-10-21T07:00:00Z") == 0  # même barre : rien d'ajouté
    before = store.path.read_bytes()
    revised = pd.DataFrame([_bar("2026-10-20", 0, NOV, 3512.0, 12), _bar("2026-10-21", 0, NOV, 3520.0, 8)])
    assert store.append(revised, fetched_at="2026-10-21T22:30:00Z") == 2
    # Le fichier ne fait que grandir : les lignes déjà écrites restent telles quelles.
    assert store.path.read_bytes().startswith(before)
    assert len(store.versions()) == 3

    then = store.as_known_at("2026-10-21T12:00:00Z")
    assert then["close"].tolist() == [3500.0]
    now = store.as_known_at()
    assert now["close"].tolist() == [3512.0, 3520.0]
    assert store.as_known_at("2026-10-20T22:00:00Z").empty
    assert set(now["fetched_at"]) == {"2026-10-21T22:30:00.000000Z"}


def test_a_revision_with_a_missing_value_is_not_filled_from_the_older_version(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    store.append(pd.DataFrame([_bar("2026-10-20", 0, NOV, 3500.0, 10)]), fetched_at="2026-10-20T22:30:00Z")
    blank = _bar("2026-10-20", 0, NOV, 3501.0, 10)
    blank["volume"] = None
    store.append(pd.DataFrame([blank]), fetched_at="2026-10-21T22:30:00Z")
    latest = store.as_known_at()
    assert latest["close"].tolist() == [3501.0]
    assert pd.isna(latest["volume"].iloc[0])


# --------------------------------------------------------------------------- série sans saut


def test_the_series_holds_one_contract_and_names_it(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    store.append(_history().reset_index(), fetched_at="2026-10-22T06:00:00Z")
    series = robusta_roll.build_series(store)
    held = series.set_index(series["date"].dt.strftime("%Y-%m-%d"))
    assert held.loc["2026-10-13", "symbol"] == "RC.c.0" and held.loc["2026-10-13", "instrument_id"] == NOV
    assert held.loc["2026-10-14", "symbol"] == "RC.c.1" and bool(held.loc["2026-10-14", "roll"])
    assert int(series["roll"].sum()) == 1
    # NOV et JAN montent de 1 $ par jour : aucun saut de 100 $ au changement d'échéance.
    assert series["price"].diff().dropna().abs().max() < 1.5
    assert series["price"].iloc[-1] == pytest.approx(series["raw_close"].iloc[-1])
    # La barre hors marché (publisher 84) n'est jamais retenue.
    assert series["raw_close"].max() < 4000


def test_the_series_never_reaches_past_the_last_complete_day_and_replays_old_knowledge(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    store.append(_history(end="2026-10-16").reset_index(), fetched_at="2026-10-17T07:00:00Z")
    store.append(_history(start="2026-10-19", partial_day="2026-10-22").reset_index(), fetched_at="2026-10-22T06:00:00Z")
    complete = robusta_roll.build_series(store, until="2026-10-21")
    assert complete["date"].max() == pd.Timestamp("2026-10-21")
    known_then = robusta_roll.build_series(store, as_of="2026-10-18T00:00:00Z")
    assert known_then["date"].max() == pd.Timestamp("2026-10-16")


def test_incremental_start_is_one_week_before_the_last_bar(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    assert robusta_roll.incremental_start(store) is None
    store.append(_history(end="2026-10-16").reset_index(), fetched_at="2026-10-17T07:00:00Z")
    assert robusta_roll.incremental_start(store) == "2026-10-09"


def test_a_request_above_its_cap_is_never_sent(tmp_path):
    store = VersionedBarStore(tmp_path / "raw.csv")
    client = FakeDatabento(_history(), cost=1.3)
    with pytest.raises(robusta_roll.CostRefused):
        robusta_roll.fetch_into_store(store, client, "2019-10-01", "2026-10-21T22:04:00", cap=robusta_roll.HISTORY_COST_CAP_USD)
    assert client.get_range_calls == [] and len(client.cost_calls) == 1
    assert not store.path.exists() and not store.fetch_log.exists()

    cheap = FakeDatabento(_history(), cost=1.226)
    entry = robusta_roll.fetch_into_store(
        store, cheap, "2026-09-01", "2026-10-21T22:04:00", cap=robusta_roll.HISTORY_COST_CAP_USD, kind="history"
    )
    assert entry["new_versions"] == len(_history()) and entry["cost_usd"] == 1.226
    call = cheap.get_range_calls[0]
    assert call["dataset"] == "IFEU.IMPACT" and call["schema"] == "ohlcv-1d" and call["stype_in"] == "continuous"
    assert call["symbols"] == ["RC.c.0", "RC.c.1", "RC.c.2"]
    assert store.fetches()[0]["kind"] == "history"


# --------------------------------------------------------------------------- collecteur


def test_the_collector_fills_missing_sessions_up_to_the_last_complete_day(tmp_path, frozen_today):
    from collect_coffee_robusta_price import EXIT_OK, main

    paths = _paths(tmp_path)
    _seed_store(paths)
    # La base a une séance sur deux depuis le 01/10 (Investing.com), rien après le 14/10.
    present = [d.strftime("%Y-%m-%d") for d in _sessions("2026-09-01", "2026-09-30")]
    present += [d.strftime("%Y-%m-%d") for d in _sessions("2026-10-01", "2026-10-14")[::2]]
    db = FakeSupabase(present)
    client = FakeDatabento(_history(partial_day="2026-10-22"))
    code = main([], client=client, supabase=db, paths=paths, now=NOW)
    assert code == EXIT_OK
    call = client.get_range_calls[0]
    assert call["start"] == "2026-10-09" and call["end"] == "2026-10-21T22:04:00"
    written = [r["date"] for r in db.inserted]
    assert written and max(written) == "2026-10-21" and "2026-10-22" not in written
    assert not set(written) & set(present)  # une ligne déjà en base n'est jamais réécrite
    assert {r["source"] for r in db.inserted} == {"databento"}
    last = next(r for r in db.inserted if r["date"] == "2026-10-21")
    assert last["symbol"] == "RC.c.1"
    series = pd.read_csv(paths.series)
    assert last["price"] == pytest.approx(series["raw_close"].iloc[-1])
    assert json.loads((paths.status_dir / "freshness_coffee_robusta.json").read_text(encoding="utf-8"))["status"] == "ok"
    journal = json.loads(next(paths.logs_dir.glob("databento_robusta_collection_*.json")).read_text(encoding="utf-8"))
    attempt = journal[-1]["attempts"][0]
    assert attempt["last_complete_day"] == "2026-10-21" and attempt["available_end"] == "2026-10-21T22:05:00Z"
    assert attempt["cost_usd"] == 0.004


def test_the_dry_run_sends_no_paid_request_and_writes_nothing(tmp_path, frozen_today):
    from collect_coffee_robusta_price import main

    paths = _paths(tmp_path)
    _seed_store(paths)
    before = paths.raw.read_bytes()
    db = FakeSupabase([d.strftime("%Y-%m-%d") for d in _sessions("2026-09-01", "2026-10-14")])
    client = FakeDatabento(_history())
    main(["--dry-run"], client=client, supabase=db, paths=paths, now=NOW)
    assert client.get_range_calls == [] and len(client.cost_calls) == 1
    assert db.inserted == []
    assert paths.raw.read_bytes() == before and not paths.raw.with_name(paths.raw.name + ".fetches.jsonl").exists()
    assert not paths.series.exists() and not paths.status_dir.exists() and not paths.logs_dir.exists()


def test_without_history_or_above_the_daily_cap_nothing_is_requested(tmp_path, frozen_today):
    from collect_coffee_robusta_price import EXIT_SOURCE_UNAVAILABLE, main

    paths = _paths(tmp_path)
    db = FakeSupabase([d.strftime("%Y-%m-%d") for d in _sessions("2026-09-01", "2026-10-14")])
    client = FakeDatabento(_history())
    assert main([], client=client, supabase=db, paths=paths, now=NOW) == EXIT_SOURCE_UNAVAILABLE
    assert client.get_range_calls == [] and db.inserted == []

    _seed_store(paths)
    expensive = FakeDatabento(_history(), cost=0.2)
    assert main([], client=expensive, supabase=db, paths=paths, now=NOW) == EXIT_SOURCE_UNAVAILABLE
    assert expensive.get_range_calls == []
    # Le brut local reste utilisable : les séances déjà connues comblent la base, le retard reste signalé.
    assert max(r["date"] for r in db.inserted) == "2026-10-16"


def test_a_closed_day_bar_is_quarantined_and_never_written(tmp_path, frozen_today, monkeypatch):
    import collect_coffee_robusta_price as collector
    from src.models import hybrid_features

    paths = _paths(tmp_path)
    _seed_store(paths)
    bars = _history()  # Databento cote le 21/10...
    db = FakeSupabase([d.strftime("%Y-%m-%d") for d in _sessions("2026-09-01", "2026-10-16")])
    real = hybrid_features.is_closed_collection_date
    # ... que le calendrier du projet déclare fermé.
    monkeypatch.setattr(
        hybrid_features, "is_closed_collection_date", lambda day, holidays=None: str(day)[:10] == "2026-10-21" or real(day)
    )
    code = collector.main([], client=FakeDatabento(bars), supabase=db, paths=paths, now=NOW)
    assert code == collector.EXIT_QUARANTINE
    assert "2026-10-21" not in {r["date"] for r in db.inserted}
    records = [json.loads(line) for line in paths.quarantine.read_text(encoding="utf-8").splitlines()]
    assert [(r["session_date"], r["reason"]) for r in records] == [("2026-10-21", "jour_ferme")]
    collector.main([], client=FakeDatabento(bars), supabase=db, paths=paths, now=NOW)
    assert len(paths.quarantine.read_text(encoding="utf-8").splitlines()) == 1  # pas de doublon chaque soir


def test_a_failed_write_is_a_real_failure(tmp_path, frozen_today):
    from collect_coffee_robusta_price import EXIT_FAILED, main

    paths = _paths(tmp_path)
    _seed_store(paths)
    db = FakeSupabase([d.strftime("%Y-%m-%d") for d in _sessions("2026-09-01", "2026-10-14")])
    db.fail_insert = True
    assert main([], client=FakeDatabento(_history()), supabase=db, paths=paths, now=NOW) == EXIT_FAILED


def test_out_of_bounds_prices_are_quarantined():
    from collect_coffee_robusta_price import plan_rows

    series = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"]),
            "raw_close": [3500.0, 9500.0, 3510.0],
            "symbol": ["RC.c.1"] * 3,
        }
    )
    inserts, quarantined = plan_rows(series, {"2026-10-19"}, until="2026-10-20", price_bounds=(1000.0, 8000.0))
    assert inserts == []
    assert [(q["session_date"], q["reason"]) for q in quarantined] == [("2026-10-20", "hors_bornes")]


# --------------------------------------------------------------------------- service et traitement


def test_robusta_preparation_uses_its_own_series_and_the_check_reads_it(tmp_path):
    from scripts.check_roll_series import SERIES, uncovered_sessions
    from src.data_collection.cocoa_roll import SERIES_PATH as COCOA_SERIES
    from src.models.release_calendar import MARKET_CALENDAR

    assert SERIES["coffee_robusta"] == robusta_roll.SERIES_PATH != COCOA_SERIES
    assert MARKET_CALENDAR["coffee_robusta"] == "trading_sessions"
    series = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-21"])})
    snapshot = pd.DataFrame({"date": ["2026-10-20", "2026-10-21", "2026-10-22"]})
    assert uncovered_sessions(snapshot, series) == ["2026-10-22"]


def test_the_full_deploy_ships_the_modules_robusta_preparation_imports():
    deploy = (ROOT / "deploy_models.ps1").read_text(encoding="utf-8")
    assert "src\\data_collection\\robusta_roll.py" in deploy
    assert "src\\data_collection\\versioned_store.py" in deploy
