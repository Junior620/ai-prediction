"""
Predicteur de courbe a terme cacao (contrats ICE).

Entrainement: historique Yahoo Finance par symbole (CCU26.NYB, ...).
Prediction: J+1 / J+7 / J+14 / J+30 par contrat actif de la derniere courbe Investing/Yahoo.
Fallback: deplacement parallele de la courbe via le % change du modele spot cacao.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models/futures")
DEFAULT_LONDON_MODELS_DIR = Path("models/futures_london")
DEFAULT_LONDON_NAMED_MODELS_DIR = Path("models/futures_london_named")
DEFAULT_HORIZONS = (1, 7, 14, 30)
LONDON_RANK_LABELS = {
    0: "Front (C.v.0)",
    1: "2e echeance",
    2: "3e echeance",
    3: "4e echeance",
}
FEATURE_COLS = [
    "lag_1",
    "lag_3",
    "lag_7",
    "ma_7",
    "ma_14",
    "ma_30",
    "change_1d",
    "change_7d",
    "vol_14",
]

# Mapping Investing symbol (CCZ26) -> Yahoo (CCZ26.NYB)
MONTH_CODES = {
    "F": "Jan",
    "G": "Feb",
    "H": "Mar",
    "J": "Apr",
    "K": "May",
    "M": "Jun",
    "N": "Jul",
    "Q": "Aug",
    "U": "Sep",
    "V": "Oct",
    "X": "Nov",
    "Z": "Dec",
}
LABEL_TO_MONTH = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def investing_to_yahoo(symbol: str) -> Optional[str]:
    """CCZ26 / CCY00 -> CCZ26.NYB (Cash skipped)."""
    sym = (symbol or "").strip().upper()
    if not sym or sym in ("CCY00", "CASH"):
        return None
    if sym.endswith(".NYB"):
        return sym
    if len(sym) >= 4 and sym.startswith("CC"):
        return f"{sym}.NYB"
    return None


def yahoo_to_label(yahoo_symbol: str) -> str:
    sym = yahoo_symbol.replace(".NYB", "")
    if len(sym) >= 5 and sym[2] in MONTH_CODES:
        month = MONTH_CODES[sym[2]]
        yy = sym[3:5]
        return f"{month} {yy}"
    return sym


def london_rank_key(rank: int) -> str:
    return f"LONDON_C.v.{int(rank)}"


def named_model_key(label: str) -> str:
    """DEC26 -> NAMED_DEC26 (cle stable pour fichiers modeles)."""
    lab = (label or "").strip().upper().replace(" ", "")
    if lab.startswith("NAMED_"):
        return lab
    return f"NAMED_{lab}"


def parse_named_label(label: str) -> Optional[Tuple[int, int]]:
    """DEC26 / NAMED_DEC26 -> (month, year) ou None."""
    import re

    text = (label or "").strip().upper().replace("NAMED_", "").replace(" ", "")
    m = re.fullmatch(r"([A-Z]{3})(\d{2})", text)
    if not m:
        return None
    month = LABEL_TO_MONTH.get(m.group(1))
    if not month:
        return None
    yy = int(m.group(2))
    year = 2000 + yy if yy < 80 else 1900 + yy
    return month, year


def _build_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": close.astype(float)})
    df["lag_1"] = df["price"].shift(1)
    df["lag_3"] = df["price"].shift(3)
    df["lag_7"] = df["price"].shift(7)
    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_14"] = df["price"].rolling(14).mean()
    df["ma_30"] = df["price"].rolling(30).mean()
    df["change_1d"] = df["price"].pct_change(1)
    df["change_7d"] = df["price"].pct_change(7)
    df["vol_14"] = df["change_1d"].rolling(14).std()
    return df


def fetch_yahoo_history(yahoo_symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(yahoo_symbol).history(period=period)
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "date", "Close": "price"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out[["date", "price"]].dropna().sort_values("date")
    return out


def fetch_london_rank_history(
    supabase_client: Any,
    rank: int,
    table: str = "cocoa_london_contracts",
) -> pd.DataFrame:
    """Historique close GBP par rang C.v.* depuis Supabase."""
    resp = (
        supabase_client.table(table)
        .select("date, close")
        .eq("contract_rank", int(rank))
        .order("date")
        .limit(5000)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    out = pd.DataFrame(resp.data)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.rename(columns={"close": "price"})
    out = out[["date", "price"]].dropna().sort_values("date")
    return out


@dataclass
class FuturesCurvePredictor:
    models_dir: Path = DEFAULT_MODELS_DIR
    horizons: Tuple[int, ...] = DEFAULT_HORIZONS
    source: str = "investing"  # investing | london | london_named

    def __post_init__(self) -> None:
        if self.source == "london" and self.models_dir == DEFAULT_MODELS_DIR:
            self.models_dir = DEFAULT_LONDON_MODELS_DIR
        elif self.source == "london_named" and self.models_dir == DEFAULT_MODELS_DIR:
            self.models_dir = DEFAULT_LONDON_NAMED_MODELS_DIR
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[int, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._london_history_cache: Dict[str, pd.DataFrame] = {}
        self._load()

    def _meta_path(self) -> Path:
        return self.models_dir / "model_info_futures.json"

    def _history_path(self, key: str) -> Path:
        import re

        safe = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")
        return self.models_dir / f"hist_{safe}.pkl"

    def _save_history(self, key: str, hist: pd.DataFrame) -> None:
        if hist is None or hist.empty:
            return
        try:
            with open(self._history_path(key), "wb") as f:
                pickle.dump(hist, f)
        except Exception as exc:
            logger.warning("Save history %s failed: %s", key, exc)

    def _load_history(self, key: str) -> Optional[pd.DataFrame]:
        path = self._history_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                hist = pickle.load(f)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                return hist
        except Exception as exc:
            logger.warning("Load history %s failed: %s", key, exc)
        return None

    def _model_path(self, yahoo_symbol: str, horizon: int) -> Path:
        import re

        safe = re.sub(r"[^A-Za-z0-9]+", "_", yahoo_symbol).strip("_")
        return self.models_dir / f"xgb_{safe}_h{horizon}.pkl"

    @staticmethod
    def _parse_model_filename(stem: str) -> Optional[Tuple[str, int]]:
        # xgb_CCZ26_NYB_h7 -> (CCZ26.NYB, 7)
        # xgb_LONDON_C_v_0_h7 -> (LONDON_C.v.0, 7)
        # xgb_NAMED_DEC26_h7 -> (NAMED_DEC26, 7)
        if not stem.startswith("xgb_") or "_h" not in stem:
            return None
        body = stem[len("xgb_") :]
        sym_part, h_part = body.rsplit("_h", 1)
        try:
            horizon = int(h_part)
        except ValueError:
            return None
        if sym_part.startswith("NAMED_"):
            return sym_part, horizon
        if sym_part.startswith("LONDON_C_v_"):
            rank = sym_part.replace("LONDON_C_v_", "")
            return london_rank_key(int(rank)), horizon
        if sym_part.endswith("_NYB"):
            yahoo = sym_part[: -len("_NYB")] + ".NYB"
        else:
            yahoo = sym_part.replace("_", ".")
        return yahoo, horizon

    def _load(self) -> None:
        meta_path = self._meta_path()
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Futures meta load failed: %s", exc)
                self._meta = {}

        rebuilt: Dict[str, Dict[int, Any]] = {}
        for pkl in self.models_dir.glob("xgb_*_h*.pkl"):
            parsed = self._parse_model_filename(pkl.stem)
            if not parsed:
                continue
            yahoo, horizon = parsed
            try:
                with open(pkl, "rb") as f:
                    rebuilt.setdefault(yahoo, {})[horizon] = pickle.load(f)
            except Exception as exc:
                logger.warning("Skip futures model %s: %s", pkl.name, exc)
        self._models = rebuilt
        # Recharge historiques locaux (evite Databento a l'inference Docker)
        for key in self._models:
            hist = self._load_history(key)
            if hist is not None:
                self._london_history_cache[key] = hist
        logger.info(
            "FuturesCurvePredictor loaded (%s): %d symbols, meta=%s",
            self.source,
            len(self._models),
            bool(self._meta),
        )

    def train_symbol(
        self,
        yahoo_symbol: str,
        history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        from xgboost import XGBRegressor

        df = history if history is not None else fetch_yahoo_history(yahoo_symbol)
        min_hist = 60 if str(yahoo_symbol).startswith("NAMED_") else 80
        if df is None or len(df) < min_hist:
            return {"symbol": yahoo_symbol, "ok": False, "reason": "insufficient_history"}

        feat = _build_features(df.set_index("date")["price"])
        metrics: Dict[str, Any] = {"symbol": yahoo_symbol, "ok": True, "horizons": {}}

        for h in self.horizons:
            target = feat["price"].shift(-h)
            train = feat[FEATURE_COLS].copy()
            train["y"] = target
            train = train.dropna()
            min_rows = 40 if str(yahoo_symbol).startswith("NAMED_") else 50
            if len(train) < min_rows:
                metrics["horizons"][str(h)] = {"ok": False, "reason": "too_few_rows"}
                continue

            split = int(len(train) * 0.8)
            X_train, y_train = train.iloc[:split][FEATURE_COLS], train.iloc[:split]["y"]
            X_val, y_val = train.iloc[split:][FEATURE_COLS], train.iloc[split:]["y"]

            model = XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mape = float(np.mean(np.abs((y_val - preds) / y_val)) * 100)
            mae = float(np.mean(np.abs(y_val - preds)))

            path = self._model_path(yahoo_symbol, h)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            self._models.setdefault(yahoo_symbol, {})[h] = model
            metrics["horizons"][str(h)] = {
                "ok": True,
                "mape": round(mape, 2),
                "mae": round(mae, 2),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "path": str(path),
            }

        return metrics

    def train_many(self, yahoo_symbols: List[str]) -> Dict[str, Any]:
        results = []
        for sym in yahoo_symbols:
            logger.info("Training futures model %s...", sym)
            try:
                results.append(self.train_symbol(sym))
            except Exception as exc:
                logger.exception("Train failed for %s", sym)
                results.append({"symbol": sym, "ok": False, "reason": str(exc)})

        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "unit": "USD/MT" if self.source != "london" else "GBP/MT",
            "horizons": list(self.horizons),
            "feature_cols": FEATURE_COLS,
            "results": results,
            "n_ok": sum(1 for r in results if r.get("ok")),
        }
        self._meta = meta
        self._meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def train_london_ranks(
        self,
        supabase_client: Any,
        ranks: Tuple[int, ...] = (0, 1, 2, 3),
    ) -> Dict[str, Any]:
        """Entrainement courbe ICE London depuis cocoa_london_contracts (GBP)."""
        self.source = "london"
        results = []
        for rank in ranks:
            key = london_rank_key(rank)
            hist = fetch_london_rank_history(supabase_client, rank)
            self._london_history_cache[key] = hist
            logger.info("Training London futures rank %s (%d rows)...", rank, len(hist))
            try:
                results.append(self.train_symbol(key, history=hist))
            except Exception as exc:
                logger.exception("Train failed for %s", key)
                results.append({"symbol": key, "ok": False, "reason": str(exc)})

        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source": "london",
            "unit": "GBP/MT",
            "venue": "ICE London",
            "ranks": list(ranks),
            "horizons": list(self.horizons),
            "feature_cols": FEATURE_COLS,
            "results": results,
            "n_ok": sum(1 for r in results if r.get("ok")),
        }
        self._meta = meta
        self._meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def train_london_named(
        self,
        count: int = 8,
        lookback_days: int = 900,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Entrainement XGBoost par echeance nommee (DEC26…) via historique Databento."""
        from src.data_collection.databento_london_collector import (
            databento_raw_symbol,
            fetch_named_symbol_history,
            upcoming_named_contracts,
        )

        self.source = "london_named"
        if self.models_dir == DEFAULT_LONDON_MODELS_DIR or self.models_dir == DEFAULT_MODELS_DIR:
            self.models_dir = DEFAULT_LONDON_NAMED_MODELS_DIR
            self.models_dir.mkdir(parents=True, exist_ok=True)

        specs = upcoming_named_contracts(count=count)
        results = []
        for spec in specs:
            label = spec["contract"]
            raw_sym = spec["symbol"]
            key = named_model_key(label)
            hist = fetch_named_symbol_history(
                raw_sym, lookback_days=lookback_days, api_key=api_key
            )
            self._london_history_cache[key] = hist
            self._save_history(key, hist)
            logger.info(
                "Training London named %s (%s, %d rows)...",
                label,
                raw_sym,
                len(hist),
            )
            try:
                metrics = self.train_symbol(key, history=hist)
                metrics["label"] = label
                metrics["databento_symbol"] = raw_sym
                results.append(metrics)
            except Exception as exc:
                logger.exception("Train failed for %s", label)
                results.append(
                    {
                        "symbol": key,
                        "label": label,
                        "ok": False,
                        "reason": str(exc),
                    }
                )

        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source": "london_named",
            "unit": "GBP/MT",
            "venue": "ICE London",
            "horizons": list(self.horizons),
            "feature_cols": FEATURE_COLS,
            "results": results,
            "n_ok": sum(1 for r in results if r.get("ok")),
            "contracts": [r.get("label") for r in results if r.get("ok")],
        }
        self._meta = meta
        self._meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def _latest_feature_row(
        self,
        yahoo_symbol: str,
        history: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        if history is not None:
            hist = history
        elif yahoo_symbol.startswith("LONDON_") or yahoo_symbol.startswith("NAMED_"):
            hist = self._london_history_cache.get(yahoo_symbol)
            if hist is None or hist.empty:
                return None
        else:
            hist = fetch_yahoo_history(yahoo_symbol, period="6mo")
        if hist is None or hist.empty:
            return None
        feat = _build_features(hist.set_index("date")["price"]).dropna()
        if feat.empty:
            return None
        return feat.iloc[-1]

    def predict_contract(
        self,
        yahoo_symbol: str,
        current_price: float,
        spot_pct_by_horizon: Optional[Dict[int, float]] = None,
        history: Optional[pd.DataFrame] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Return {horizon: {price, method, mape?}}."""
        out: Dict[int, Dict[str, Any]] = {}
        models = self._models.get(yahoo_symbol, {})
        row = None
        if models:
            try:
                row = self._latest_feature_row(yahoo_symbol, history=history)
            except Exception as exc:
                logger.warning("Feature row failed for %s: %s", yahoo_symbol, exc)

        for h in self.horizons:
            method = "spot_shift"
            price = None
            if h in models and row is not None:
                try:
                    X = row[FEATURE_COLS].values.reshape(1, -1)
                    price = float(models[h].predict(X)[0])
                    method = "xgboost"
                except Exception as exc:
                    logger.warning("Predict %s h=%s failed: %s", yahoo_symbol, h, exc)
                    price = None

            if price is None:
                pct = (spot_pct_by_horizon or {}).get(h, 0.0)
                price = float(current_price) * (1.0 + pct)
                method = "spot_shift"

            # Bound crazy jumps (>25% from current)
            if current_price > 0:
                lo, hi = current_price * 0.75, current_price * 1.25
                price = float(min(max(price, lo), hi))

            out[h] = {"price": round(price, 2), "method": method}
        return out

    def predict_curve(
        self,
        contracts: List[Dict[str, Any]],
        spot_pct_by_horizon: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        contracts: list with keys contract/symbol/price_usd (Investing or Yahoo format).
        Returns enriched list with predictions[{horizon, price, method}].
        """
        enriched: List[Dict[str, Any]] = []
        for c in contracts:
            symbol = str(c.get("symbol") or "")
            label = str(c.get("contract") or symbol)
            price = float(c.get("price_usd") or c.get("price") or c.get("close") or 0)
            yahoo = investing_to_yahoo(symbol)
            item: Dict[str, Any] = {
                "contract": label,
                "symbol": symbol,
                "yahoo_symbol": yahoo,
                "price_usd": price,
                "change": c.get("change"),
                "volume": c.get("volume"),
                "predictions": [],
            }
            if price <= 0:
                enriched.append(item)
                continue

            if yahoo:
                preds = self.predict_contract(yahoo, price, spot_pct_by_horizon)
            else:
                # Cash / non-yahoo: spot shift only
                preds = {
                    h: {
                        "price": round(price * (1.0 + (spot_pct_by_horizon or {}).get(h, 0.0)), 2),
                        "method": "spot_shift",
                    }
                    for h in self.horizons
                }

            for h, p in preds.items():
                item["predictions"].append(
                    {
                        "horizon": h,
                        "price": p["price"],
                        "method": p["method"],
                        "change_pct": round((p["price"] / price - 1.0) * 100, 2) if price else None,
                    }
                )
            enriched.append(item)
        return enriched

    def predict_london_curve(
        self,
        contracts: List[Dict[str, Any]],
        spot_pct_by_horizon: Optional[Dict[int, float]] = None,
        history_by_rank: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> List[Dict[str, Any]]:
        """contracts: rows with contract_rank, symbol, close (GBP)."""
        enriched: List[Dict[str, Any]] = []
        for c in contracts:
            rank = int(c.get("contract_rank", 0))
            symbol = str(c.get("symbol") or f"C.v.{rank}")
            label = str(c.get("label") or LONDON_RANK_LABELS.get(rank, symbol))
            price = float(c.get("close") or c.get("price") or c.get("price_usd") or 0)
            key = london_rank_key(rank)
            hist = (history_by_rank or {}).get(rank)
            item: Dict[str, Any] = {
                "contract": label,
                "symbol": symbol,
                "yahoo_symbol": key,
                "price_usd": price,  # GBP when source=london
                "change": c.get("change"),
                "volume": c.get("volume"),
                "predictions": [],
            }
            if price <= 0:
                enriched.append(item)
                continue

            preds = self.predict_contract(
                key, price, spot_pct_by_horizon, history=hist
            )
            for h, p in preds.items():
                item["predictions"].append(
                    {
                        "horizon": h,
                        "price": p["price"],
                        "method": p["method"],
                        "change_pct": round((p["price"] / price - 1.0) * 100, 2) if price else None,
                    }
                )
            enriched.append(item)
        return enriched

    def predict_london_named_curve(
        self,
        contracts: List[Dict[str, Any]],
        spot_pct_by_horizon: Optional[Dict[int, float]] = None,
        history_by_key: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Dict[str, Any]]:
        """contracts: mois nommes Londres (contract=DEC26, price_usd=GBP)."""
        from src.data_collection.databento_london_collector import (
            databento_raw_symbol,
            fetch_named_symbol_history,
        )

        enriched: List[Dict[str, Any]] = []
        for c in contracts:
            raw_label = str(c.get("contract") or c.get("symbol") or "").strip().upper()
            parsed = parse_named_label(raw_label) or parse_named_label(
                str(c.get("symbol") or "")
            )
            if parsed:
                month_name = next(
                    (k for k, v in LABEL_TO_MONTH.items() if v == parsed[0]),
                    "DEC",
                )
                label = f"{month_name}{str(parsed[1])[2:]}"
            else:
                label = raw_label.replace(" ", "")

            key = named_model_key(label)
            price = float(c.get("price_usd") or c.get("price") or c.get("close") or 0)
            hist = (history_by_key or {}).get(key)
            if hist is None and key not in self._london_history_cache and parsed:
                try:
                    raw = databento_raw_symbol(parsed[0], parsed[1])
                    hist = fetch_named_symbol_history(raw, lookback_days=200)
                    self._london_history_cache[key] = hist
                except Exception as exc:
                    logger.warning("Named history fetch %s: %s", label, exc)

            item: Dict[str, Any] = {
                "contract": label,
                "symbol": str(c.get("symbol") or label),
                "yahoo_symbol": key,
                "price_usd": price,
                "change": c.get("change"),
                "volume": c.get("volume"),
                "predictions": [],
            }
            if price <= 0:
                enriched.append(item)
                continue

            preds = self.predict_contract(
                key,
                price,
                spot_pct_by_horizon,
                history=hist
                if hist is not None
                else self._london_history_cache.get(key),
            )
            for h, p in preds.items():
                item["predictions"].append(
                    {
                        "horizon": h,
                        "price": p["price"],
                        "method": p["method"],
                        "change_pct": round((p["price"] / price - 1.0) * 100, 2)
                        if price
                        else None,
                    }
                )
            enriched.append(item)
        return enriched
