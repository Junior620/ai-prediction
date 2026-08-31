"""
Collecteur Databento — London Cocoa (IFEU.IMPACT, continuous C.v.*).

Schemas :
- ohlcv-1d  → Open/High/Low/Close/Volume
- statistics (stat_type=9) → Open Interest

Pas de flux live : historique + barre quotidienne uniquement.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DATASET = "IFEU.IMPACT"
PARENT_SYMBOL = "C.FUT"
DEFAULT_CONTINUOUS = "C.v.0"
DEFAULT_RANKS = (0, 1, 2, 3)
# Databento StatType.OPEN_INTEREST
STAT_TYPE_OPEN_INTEREST = 9
SOURCE = "databento"


@dataclass
class DatabentoBar:
    date: str
    price: float  # close £/T
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    symbol: str = DEFAULT_CONTINUOUS
    source: str = SOURCE
    contract_rank: int = 0


@dataclass
class DatabentoFetchResult:
    bars: List[DatabentoBar] = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def latest(self) -> Optional[DatabentoBar]:
        return self.bars[-1] if self.bars else None


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    key = (explicit or os.getenv("DATABENTO_API_KEY") or "").strip()
    return key or None


def _make_client(api_key: Optional[str] = None):
    key = get_api_key(api_key)
    if not key:
        return None
    try:
        import databento as db
    except ImportError:
        logger.error("Package databento non installe")
        return None
    return db.Historical(key)


def continuous_symbols(ranks: Sequence[int] = DEFAULT_RANKS) -> List[str]:
    return [f"C.v.{r}" for r in ranks]


def _to_date_str(ts) -> str:
    if isinstance(ts, str):
        return ts[:10]
    if hasattr(ts, "strftime"):
        # Databento index is often tz-aware UTC end-of-bar
        try:
            return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d")
        except (TypeError, AttributeError, ValueError):
            return pd.Timestamp(ts).strftime("%Y-%m-%d")
    return str(ts)[:10]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    # Databento UNDEF_PRICE sentinel
    if abs(f) > 1e14:
        return None
    return f


# IFEU.IMPACT on-exchange publisher (XOFF=84 produit des barres aberrantes)
ON_EXCHANGE_PUBLISHER_ID = 57


def _prefer_on_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """
    IFEU publie 2 barres ohlcv/jour (on-exchange + off-market XOFF).
    On garde le publisher on-exchange (57) ; sinon la barre au plus gros volume.
    """
    if df is None or df.empty:
        return df
    work = df.copy()
    if "publisher_id" not in work.columns:
        return work

    work = work.reset_index()
    ts_col = "ts_event" if "ts_event" in work.columns else work.columns[0]
    work["_date"] = pd.to_datetime(work[ts_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    work["_sym"] = work["symbol"].astype(str) if "symbol" in work.columns else DEFAULT_CONTINUOUS

    preferred_rows = []
    for (_, _), group in work.groupby(["_date", "_sym"], sort=False):
        on_ex = group[group["publisher_id"] == ON_EXCHANGE_PUBLISHER_ID]
        if not on_ex.empty:
            preferred_rows.append(on_ex.iloc[-1])
            continue
        if "volume" in group.columns:
            preferred_rows.append(group.loc[group["volume"].fillna(0).idxmax()])
        else:
            preferred_rows.append(group.iloc[-1])

    out = pd.DataFrame(preferred_rows)
    if out.empty:
        return df
    out = out.set_index(ts_col)
    return out.drop(columns=["_date", "_sym"], errors="ignore")


def _ohlcv_df_to_bars(
    df: pd.DataFrame,
    price_bounds: Tuple[float, float],
    default_rank: int = 0,
) -> List[DatabentoBar]:
    """Mappe un DataFrame ohlcv-1d Databento vers des DatabentoBar."""
    if df is None or df.empty:
        return []

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "ts_event" in work.columns:
            work = work.set_index("ts_event")
        else:
            return []

    work = _prefer_on_exchange(work)

    lo, hi = price_bounds
    bars: List[DatabentoBar] = []
    for ts, row in work.iterrows():
        close = _safe_float(row.get("close"))
        if close is None or not (lo <= close <= hi):
            continue
        symbol = str(row.get("symbol") or DEFAULT_CONTINUOUS)
        rank = default_rank
        if ".v." in symbol:
            try:
                rank = int(symbol.rsplit(".", 1)[-1])
            except ValueError:
                rank = default_rank
        bars.append(
            DatabentoBar(
                date=_to_date_str(ts),
                price=close,
                open=_safe_float(row.get("open")),
                high=_safe_float(row.get("high")),
                low=_safe_float(row.get("low")),
                volume=_safe_float(row.get("volume")),
                symbol=symbol,
                contract_rank=rank,
            )
        )
    bars.sort(key=lambda b: (b.date, b.contract_rank))
    return bars


def _parse_oi_df(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Retourne {(date, symbol): open_interest} depuis statistics.
    Preferre la derniere revision par (date, symbol).
    """
    if df is None or df.empty:
        return {}

    work = df.copy()
    if "stat_type" in work.columns:
        work = work[work["stat_type"] == STAT_TYPE_OPEN_INTEREST].copy()
    if work.empty:
        return {}

    # Exclure les spreads (symboles avec tiret)
    if "symbol" in work.columns:
        work = work[~work["symbol"].astype(str).str.contains("-", regex=False)].copy()

    qty_col = "quantity" if "quantity" in work.columns else None
    if qty_col is None:
        for cand in ("value", "price"):
            if cand in work.columns:
                qty_col = cand
                break
    if qty_col is None:
        return {}

    # Date de reference = ts_ref si present, sinon index
    if "ts_ref" in work.columns:
        work["date"] = pd.to_datetime(work["ts_ref"], utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    else:
        work = work.reset_index()
        idx_col = "ts_event" if "ts_event" in work.columns else work.columns[0]
        work["date"] = pd.to_datetime(work[idx_col], utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    work = work.dropna(subset=["date"])
    work = work.sort_values(by=[c for c in ("ts_recv", "ts_event") if c in work.columns])
    grouped = work.groupby(["date", "symbol"], as_index=False).last()

    out: Dict[Tuple[str, str], float] = {}
    for _, row in grouped.iterrows():
        q = _safe_float(row.get(qty_col))
        if q is None:
            continue
        out[(str(row["date"]), str(row["symbol"]))] = q
    return out


def fetch_daily_bars(
    start: str,
    end: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    price_bounds: Tuple[float, float] = (1500.0, 6000.0),
    client=None,
) -> DatabentoFetchResult:
    """
    Recupere les barres ohlcv-1d pour un ou plusieurs continuous symbols.
    end inclusif cote appelant ; Databento utilise end exclusif (end+1j),
    borne automatiquement a aujourd'hui UTC.
    """
    result = DatabentoFetchResult()
    syms = list(symbols or [DEFAULT_CONTINUOUS])
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Historique Databento (sans licence live) : ~24h de delai
    hist_limit = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    end = end or hist_limit
    if end > hist_limit:
        end = hist_limit
    if start > end:
        result.error = f"start_after_end:{start}>{end}"
        result.attempts.append({"strategy": "databento_ohlcv", "ok": False, "detail": result.error})
        return result

    hist = client or _make_client(api_key)
    if hist is None:
        result.error = "missing_api_key_or_package"
        result.attempts.append({"strategy": "databento_ohlcv", "ok": False, "detail": result.error})
        return result

    # end Databento est exclusif ; borne au delai historique
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if end_exclusive > hist_limit:
        end_exclusive = hist_limit
    if start >= end_exclusive:
        end_exclusive = hist_limit
        start = (pd.Timestamp(hist_limit) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        data = hist.timeseries.get_range(
            dataset=DATASET,
            symbols=syms,
            schema="ohlcv-1d",
            stype_in="continuous",
            start=start,
            end=end_exclusive,
        )
        df = data.to_df()
        bars = _ohlcv_df_to_bars(df, price_bounds=price_bounds)
        result.bars = bars
        result.ok = len(bars) > 0
        result.attempts.append(
            {
                "strategy": "databento_ohlcv",
                "ok": result.ok,
                "symbols": syms,
                "rows": len(bars),
                "start": start,
                "end": end,
            }
        )
    except Exception as exc:
        logger.exception("Databento ohlcv-1d failed: %s", exc)
        result.error = str(exc)
        result.attempts.append(
            {"strategy": "databento_ohlcv", "ok": False, "detail": str(exc)}
        )
    return result


def fetch_open_interest(
    start: str,
    end: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    client=None,
) -> Tuple[Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
    """
    Open Interest via schema statistics.
    Tente d'abord continuous (C.v.*), puis parent C.FUT en fallback.
    """
    attempts: List[Dict[str, Any]] = []
    hist = client or _make_client(api_key)
    if hist is None:
        attempts.append({"strategy": "databento_oi", "ok": False, "detail": "missing_api_key"})
        return {}, attempts

    hist_limit = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    end = end or hist_limit
    if end > hist_limit:
        end = hist_limit
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if end_exclusive > hist_limit:
        end_exclusive = hist_limit
    syms = list(symbols or [DEFAULT_CONTINUOUS])

    # 1) Continuous
    try:
        data = hist.timeseries.get_range(
            dataset=DATASET,
            symbols=syms,
            schema="statistics",
            stype_in="continuous",
            start=start,
            end=end_exclusive,
        )
        oi_map = _parse_oi_df(data.to_df())
        attempts.append(
            {
                "strategy": "databento_oi_continuous",
                "ok": bool(oi_map),
                "rows": len(oi_map),
            }
        )
        if oi_map:
            return oi_map, attempts
    except Exception as exc:
        logger.warning("OI continuous failed: %s", exc)
        attempts.append({"strategy": "databento_oi_continuous", "ok": False, "detail": str(exc)})

    # 2) Parent C.FUT — OI par contrat brut (utile pour agregats)
    try:
        data = hist.timeseries.get_range(
            dataset=DATASET,
            symbols=[PARENT_SYMBOL],
            schema="statistics",
            stype_in="parent",
            start=start,
            end=end_exclusive,
        )
        oi_map = _parse_oi_df(data.to_df())
        attempts.append(
            {
                "strategy": "databento_oi_parent",
                "ok": bool(oi_map),
                "rows": len(oi_map),
            }
        )
        return oi_map, attempts
    except Exception as exc:
        logger.warning("OI parent failed: %s", exc)
        attempts.append({"strategy": "databento_oi_parent", "ok": False, "detail": str(exc)})
        return {}, attempts


def attach_open_interest(
    bars: List[DatabentoBar],
    oi_map: Dict[Tuple[str, str], float],
) -> List[DatabentoBar]:
    """Joint OI aux barres par (date, symbol). Fallback : OI max du jour (parent)."""
    if not oi_map:
        return bars

    by_date_max: Dict[str, float] = {}
    for (d, _sym), qty in oi_map.items():
        by_date_max[d] = max(by_date_max.get(d, 0.0), qty)

    for bar in bars:
        key = (bar.date, bar.symbol)
        if key in oi_map:
            bar.open_interest = oi_map[key]
        elif bar.date in by_date_max and bar.contract_rank == 0:
            # Front month : prendre le max OI du jour si continuous n'a pas matche
            bar.open_interest = by_date_max[bar.date]
    return bars


def fetch_daily_bars_with_oi(
    start: str,
    end: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    price_bounds: Tuple[float, float] = (1500.0, 6000.0),
    include_oi: bool = True,
    client=None,
) -> DatabentoFetchResult:
    """OHLCV + Open Interest joints."""
    result = fetch_daily_bars(
        start=start,
        end=end,
        symbols=symbols,
        api_key=api_key,
        price_bounds=price_bounds,
        client=client,
    )
    if not result.ok or not include_oi:
        return result

    oi_map, oi_attempts = fetch_open_interest(
        start=start,
        end=end,
        symbols=symbols,
        api_key=api_key,
        client=client,
    )
    result.attempts.extend(oi_attempts)
    result.bars = attach_open_interest(result.bars, oi_map)
    return result


def fetch_latest_spot(
    api_key: Optional[str] = None,
    price_bounds: Tuple[float, float] = (1500.0, 6000.0),
    lookback_days: int = 10,
    include_oi: bool = True,
) -> DatabentoFetchResult:
    """Derniere barre disponible (front month C.v.0)."""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    result = fetch_daily_bars_with_oi(
        start=start,
        end=end,
        symbols=[DEFAULT_CONTINUOUS],
        api_key=api_key,
        price_bounds=price_bounds,
        include_oi=include_oi,
    )
    if result.bars:
        # Ne garder que la derniere date front-month
        front = [b for b in result.bars if b.contract_rank == 0]
        if front:
            latest_date = front[-1].date
            result.bars = [b for b in front if b.date == latest_date]
    return result


def bars_to_supabase_rows(
    bars: List[DatabentoBar],
    front_only: bool = True,
    include_oi: bool = True,
) -> List[Dict[str, Any]]:
    """Lignes pour cocoa_london_prices (une ligne / date, front month)."""
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for bar in bars:
        if front_only and bar.contract_rank != 0:
            continue
        if bar.date in seen:
            continue
        seen.add(bar.date)
        row: Dict[str, Any] = {
            "date": bar.date,
            "price": float(bar.price),
            "symbol": bar.symbol,
            "source": SOURCE,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
            "collected_at": datetime.now().isoformat(),
        }
        if include_oi and bar.open_interest is not None:
            row["open_interest"] = float(bar.open_interest)
        rows.append(row)
    return rows


def bars_to_contract_rows(bars: List[DatabentoBar]) -> List[Dict[str, Any]]:
    """Lignes pour cocoa_london_contracts (toutes echeances)."""
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for bar in bars:
        key = (bar.date, bar.contract_rank)
        if key in seen:
            continue
        seen.add(key)
        row: Dict[str, Any] = {
            "date": bar.date,
            "contract_rank": int(bar.contract_rank),
            "symbol": bar.symbol,
            "close": float(bar.price),
            "volume": bar.volume,
            "source": SOURCE,
            "collected_at": datetime.now().isoformat(),
        }
        if bar.open_interest is not None:
            row["open_interest"] = float(bar.open_interest)
        rows.append(row)
    return rows


def write_collection_journal(
    payload: Dict[str, Any],
    logs_dir: Optional[Path] = None,
) -> Path:
    """Ecrit logs/databento_collection_YYYYMMDD.json."""
    base = logs_dir or Path("logs")
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    path = base / f"databento_collection_{day}.json"
    existing: List[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []
    existing.append(payload)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
