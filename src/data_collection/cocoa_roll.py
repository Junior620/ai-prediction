"""Série cacao Londres sans saut de contrat.

``C.v.0`` (Databento) suit chaque jour l'échéance au plus gros volume. Quand
deux échéances se disputent ce volume, la série passe de l'une à l'autre et
l'écart entre échéances (jusqu'à 2 000 £ en 2024) devient un faux mouvement
de ±15 %.

Ici on suit une seule échéance à la fois. On ne passe à une échéance plus
lointaine (la plus échangée) que lorsque son volume dépasse celui de
l'échéance tenue, ou quand celle-ci disparaît, et on ne revient jamais en
arrière. Chaque rendement quotidien est
calculé sur un même contrat. Le niveau est ajusté par ratio : la dernière
valeur égale la clôture de l'échéance tenue, le passé est mis à l'échelle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "databento_cocoa_calendar_raw.csv"
SERIES_PATH = ROOT / "data" / "cocoa_roll_series.csv"
CALENDAR_SYMBOLS = ("C.c.0", "C.c.1", "C.c.2")
DATASET = "IFEU.IMPACT"
ON_EXCHANGE_PUBLISHER_ID = 57
_ADJUSTED = ("price", "open", "high", "low")


def fetch_calendar_bars(
    start: str,
    end: str,
    client=None,
    api_key: Optional[str] = None,
    symbols: tuple[str, ...] = CALENDAR_SYMBOLS,
) -> pd.DataFrame:
    """Barres ohlcv-1d C.c.0..2 (ou ``symbols``) avec l'identifiant du contrat. ``end`` exclusif."""
    if client is None:
        import os

        import databento as db

        key = (api_key or os.getenv("DATABENTO_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("DATABENTO_API_KEY manquante")
        client = db.Historical(key)
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=list(symbols),
        schema="ohlcv-1d",
        stype_in="continuous",
        start=start,
        end=end,
    )
    return data.to_df().reset_index()


def merge_raw(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Les barres récentes remplacent les anciennes à même date, symbole et publisher."""
    frames = [f for f in (existing, fresh) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    raw = raw.drop_duplicates(subset=["ts_event", "symbol", "publisher_id"], keep="last")
    return raw.sort_values(["ts_event", "symbol", "publisher_id"]).reset_index(drop=True)


def on_exchange_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Une barre par date et rang : publisher on-exchange, sinon la plus échangée."""
    work = raw.copy()
    work["date"] = pd.to_datetime(work["ts_event"], utc=True).dt.tz_convert(None).dt.normalize()
    work["rank"] = work["symbol"].astype(str).str.rsplit(".", n=1).str[-1].astype(int)
    work["on_exchange"] = work["publisher_id"] == ON_EXCHANGE_PUBLISHER_ID
    work = work.sort_values(["date", "rank", "on_exchange", "volume"])
    bars = work.groupby(["date", "rank"], as_index=False).last()
    return bars[["date", "rank", "instrument_id", "open", "high", "low", "close", "volume"]]


def liquid_roll_series(bars: pd.DataFrame) -> pd.DataFrame:
    """Une échéance tenue par jour, rendements sur ce même contrat, niveau ajusté par ratio."""
    if bars.empty:
        return pd.DataFrame(columns=["date", "instrument_id", "raw_close", "roll", *_ADJUSTED, "volume"])
    rows = []
    held: Optional[int] = None
    previous: dict[int, float] = {}
    for date, day in bars.sort_values(["date", "rank"]).groupby("date", sort=True):
        day = day.set_index("instrument_id")
        order = list(day.sort_values("rank").index)
        rolled = False
        if held is None:
            held = int(day["volume"].idxmax())
        elif held not in day.index:
            # Échéance expirée : la plus proche des suivantes encore cotées.
            later = [i for i in order if i in previous]
            held = int(later[0] if later else order[0])
            rolled = True
        else:
            position = order.index(held)
            later = [i for i in order[position + 1:] if day.at[i, "volume"] > day.at[held, "volume"]]
            if later:
                held = int(max(later, key=lambda i: day.at[i, "volume"]))
                rolled = True
        bar = day.loc[held]
        close = float(bar["close"])
        base = previous.get(held)
        ret = close / base - 1.0 if base and base > 0 else np.nan
        rows.append(
            {
                "date": date,
                "instrument_id": held,
                "raw_open": float(bar["open"]),
                "raw_high": float(bar["high"]),
                "raw_low": float(bar["low"]),
                "raw_close": close,
                "volume": float(bar["volume"]),
                "roll": rolled,
                "ret": ret,
            }
        )
        previous = {int(i): float(c) for i, c in day["close"].items()}
    out = pd.DataFrame(rows)
    # Le premier jour (ou un contrat jamais vu la veille) n'a pas de rendement :
    # on garde le niveau, sans saut inventé.
    growth = (1.0 + out["ret"].fillna(0.0)).cumprod()
    factor = growth / out["raw_close"]
    factor = factor * (out["raw_close"].iloc[-1] / growth.iloc[-1])
    out["price"] = growth * (out["raw_close"].iloc[-1] / growth.iloc[-1])
    for column in ("open", "high", "low"):
        out[column] = out[f"raw_{column}"] * factor
    return out[["date", "instrument_id", "raw_close", "roll", "ret", *_ADJUSTED, "volume"]]


def load_series(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    target = Path(path) if path is not None else SERIES_PATH
    if not target.exists():
        return None
    series = pd.read_csv(target, parse_dates=["date"])
    series["date"] = series["date"].dt.normalize()
    return series


def apply_roll_series(
    frame: pd.DataFrame,
    series: Optional[pd.DataFrame],
    is_closed: Optional[Callable[[pd.Timestamp], bool]] = None,
    refresh_hint: str = "scripts/build_cocoa_roll_series.py",
) -> pd.DataFrame:
    """Remplace prix et OHLC par la série sans saut. Idempotent.

    Une séance de ``frame`` absente de la série avant sa fin est écartée : la
    source n'a pas coté ce jour-là (lignes ``ice_london_bootstrap_fx``). Une
    ligne collectée un jour de fermeture (``is_closed``) reste, avec le
    dernier cours ajusté. Les dates postérieures à la série restent brutes et
    ``roll_adjusted`` vaut False ; la série doit être rafraîchie chaque jour.
    """
    if series is None or series.empty or frame is None or len(frame) == 0:
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    end = series["date"].max()
    indexed = series.set_index("date")
    out = out.sort_values("date").reset_index(drop=True)
    known = out["date"].isin(indexed.index)
    after = out["date"] > end
    closed = out["date"].map(is_closed).astype(bool) if is_closed is not None else pd.Series(False, index=out.index)
    carried = ~known & ~after & closed
    dropped = int((~known & ~after & ~closed).sum())
    if dropped:
        logger.info(f"Série sans saut : {dropped} séances sans cotation Databento écartées")
    out = out[known | after | carried].copy()
    known = out["date"].isin(indexed.index)
    carried = ~known & (out["date"] <= end)
    matched = out.loc[known, "date"]
    for column in _ADJUSTED:
        if column in indexed.columns:
            out.loc[known, column] = indexed.loc[matched, column].to_numpy()
            out.loc[carried, column] = np.nan
            out[column] = out[column].where(~carried, out[column].ffill())
    if "volume" in out.columns:
        out.loc[known, "volume"] = indexed.loc[matched, "volume"].to_numpy()
    out["roll_adjusted"] = known | carried
    pending = int((out["date"] > end).sum())
    if pending > 5:
        logger.warning(f"Série sans saut en retard de {pending} séances : relancer {refresh_hint}")
    return out.reset_index(drop=True)


def refresh(raw_path: Path = RAW_PATH, series_path: Path = SERIES_PATH, *, fetch: bool = True, client=None) -> pd.DataFrame:
    """Complète le brut depuis la dernière date (moins une semaine) et réécrit la série."""
    existing = pd.read_csv(raw_path) if raw_path.exists() else None
    if fetch:
        if existing is not None and len(existing):
            last = pd.to_datetime(existing["ts_event"], utc=True).max()
            start = (last - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            start = "2019-10-01"
        from src.data_collection.databento_london_collector import _make_client, daily_end_exclusive

        if client is None:
            client = _make_client()
            if client is None:
                raise RuntimeError("DATABENTO_API_KEY manquante")
        # Même borne que la collecte des prix : jusqu'à la dernière journée complète publiée.
        # Sinon la série s'arrêterait une séance avant la base et check_roll_series bloquerait.
        end = daily_end_exclusive(client)
        if start < end:
            fresh = fetch_calendar_bars(start, end, client=client)
            existing = merge_raw(existing, fresh)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            existing.to_csv(raw_path, index=False)
    if existing is None or not len(existing):
        raise RuntimeError(f"Aucune barre Databento dans {raw_path}")
    series = liquid_roll_series(on_exchange_bars(existing))
    series_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(series_path, index=False, date_format="%Y-%m-%d")
    return series
