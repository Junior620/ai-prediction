"""A cached study fit is reused only when its inputs are the same."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from scripts.score_causal_cocoa import (
    CacheRejected,
    _fit_session,
    accept_cached_fit,
    history_fingerprint,
    study_identity,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-18", "2026-09-01"]),
            "price": [4700.0, 4805.0],
            "open": [4700.0, 4805.0],
            "high": [4710.0, 4810.0],
            "low": [4690.0, 4800.0],
            "volume": [10.0, 11.0],
        }
    )


def _identity(**overrides) -> dict:
    payload = dict(
        snapshot_fingerprint="snapshot",
        calendar="trading_sessions",
        feature_cols=["price"],
        params={"n_estimators": 200},
        code_version="code",
        origins=["2026-09-01"],
    )
    payload.update(overrides)
    return study_identity(**payload)


def _payload(history: pd.DataFrame, identity: dict) -> dict:
    return {
        "identity": identity,
        "feature_cols": ["price"],
        "history_fingerprint": history_fingerprint(history),
        "train_end": "2026-09-01",
        "latest_training_target": "2026-08-18",
        "samples": 3,
        "training_rows": 3,
        "model": None,
        "last_row": None,
    }


def test_a_changed_input_refuses_the_cached_fit(tmp_path: Path):
    history = _history()
    identity = _identity()
    payload = _payload(history, identity)
    accept_cached_fit(
        payload,
        identity=identity,
        history_fingerprint_value=history_fingerprint(history),
        session="2026-09-01",
        feature_cols=["price"],
    )
    changed = history.copy()
    changed.loc[changed.index[-1], "price"] = 5305.0
    with pytest.raises(CacheRejected):
        accept_cached_fit(
            payload,
            identity=identity,
            history_fingerprint_value=history_fingerprint(changed),
            session="2026-09-01",
            feature_cols=["price"],
        )
    with pytest.raises(CacheRejected):
        accept_cached_fit(
            payload,
            identity=_identity(feature_cols=["other"]),
            history_fingerprint_value=history_fingerprint(history),
            session="2026-09-01",
            feature_cols=["other"],
        )
    with pytest.raises(CacheRejected):
        accept_cached_fit(
            payload,
            identity=_identity(calendar="keep_closed_rows"),
            history_fingerprint_value=history_fingerprint(history),
            session="2026-09-01",
            feature_cols=["price"],
        )
    with pytest.raises(CacheRejected):
        accept_cached_fit(
            payload,
            identity=_identity(params={"n_estimators": 20}),
            history_fingerprint_value=history_fingerprint(history),
            session="2026-09-01",
            feature_cols=["price"],
        )
    future = dict(payload)
    future["latest_training_target"] = "2026-09-02"
    with pytest.raises(CacheRejected):
        accept_cached_fit(
            future,
            identity=identity,
            history_fingerprint_value=history_fingerprint(history),
            session="2026-09-01",
            feature_cols=["price"],
        )
    bare = {key: value for key, value in payload.items() if key != "identity"}
    path = tmp_path / "fits" / "2026-09-01_h14.pkl"
    path.parent.mkdir()
    path.write_bytes(pickle.dumps(bare))
    before = path.read_bytes()
    with pytest.raises(CacheRejected):
        _fit_session(
            history,
            "2026-09-01",
            [14],
            ["price"],
            folder=tmp_path,
            identity=identity,
        )
    assert path.read_bytes() == before
