"""Stockage brut versionné : chaque version d'une barre, avec l'heure où on l'a reçue.

Le brut cacao (``merge_raw``) remplace une barre par la plus récente : une révision de
Databento efface ce qu'on savait avant. Ici rien n'est réécrit. Le fichier CSV ne fait
que grandir ; une ligne est ajoutée quand une barre apparaît ou quand ses valeurs
changent, avec son ``fetched_at`` (UTC). Une barre re-téléchargée à l'identique
n'ajoute rien : la version déjà connue reste valable.

``as_known_at(t)`` rend, pour chaque barre, la dernière version reçue au plus tard à
``t`` : les données telles qu'on les connaissait à cet instant (« point-in-time »).
Un historique rattrapé en une fois n'est connu qu'à partir de son ``fetched_at``.

Chaque téléchargement est aussi noté dans ``<fichier>.fetches.jsonl`` (plage demandée,
fin de plage publiée, coût estimé, versions ajoutées).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

KEY_COLUMNS = ("ts_event", "symbol", "publisher_id")
VALUE_COLUMNS = ("instrument_id", "rtype", "open", "high", "low", "close", "volume")
COLUMNS = (*KEY_COLUMNS, *VALUE_COLUMNS, "fetched_at")


def utc_stamp(value: Any) -> str:
    """Horodatage UTC ISO à la microseconde, triable comme du texte."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Colonnes fixes, ``ts_event`` en texte UTC : la même barre a toujours la même clé."""
    work = frame.copy()
    if "ts_event" not in work.columns:
        work = work.reset_index()
    missing = [c for c in KEY_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(f"Colonnes clés absentes : {missing}")
    work["ts_event"] = pd.to_datetime(work["ts_event"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    work["symbol"] = work["symbol"].astype(str)
    for column in VALUE_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA
    return work[[*KEY_COLUMNS, *VALUE_COLUMNS]]


def _signature(row: Mapping[str, Any]) -> tuple:
    """Valeurs comparées entre deux versions (flottants arrondis : le CSV ne les déforme pas)."""
    out = []
    for column in VALUE_COLUMNS:
        value = row.get(column)
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            out.append(None)
        else:
            try:
                out.append(round(float(value), 6))
            except (TypeError, ValueError):
                out.append(str(value))
    return tuple(out)


def _key(row: Mapping[str, Any]) -> tuple:
    return (str(row["ts_event"]), str(row["symbol"]), int(float(row["publisher_id"])))


class VersionedBarStore:
    """Barres ohlcv versionnées dans un CSV qu'on ne fait qu'allonger."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def fetch_log(self) -> Path:
        return self.path.with_name(self.path.name + ".fetches.jsonl")

    def versions(self) -> pd.DataFrame:
        """Toutes les versions, dans l'ordre d'écriture."""
        if not self.path.exists():
            return pd.DataFrame(columns=list(COLUMNS))
        return pd.read_csv(self.path, dtype={"ts_event": str, "symbol": str, "fetched_at": str})

    def as_known_at(self, when: Any = None) -> pd.DataFrame:
        """Dernière version de chaque barre reçue au plus tard à ``when`` (toutes si None)."""
        versions = self.versions()
        if when is not None and len(versions):
            versions = versions[versions["fetched_at"] <= utc_stamp(when)]
        if versions.empty:
            return versions.reset_index(drop=True)
        # Ordre d'écriture conservé : à fetched_at égal, la ligne écrite après l'emporte.
        # drop_duplicates et non groupby().last(), qui comblerait un vide par une version plus ancienne.
        latest = versions.reset_index(drop=True).sort_values("fetched_at", kind="stable")
        latest = latest.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
        return latest.sort_values(["ts_event", "symbol", "publisher_id"]).reset_index(drop=True)

    def last_event(self) -> Optional[pd.Timestamp]:
        versions = self.versions()
        if versions.empty:
            return None
        return pd.to_datetime(versions["ts_event"], utc=True).max()

    def append(self, fresh: pd.DataFrame, fetched_at: Any) -> int:
        """Ajoute les barres nouvelles ou modifiées. Rend le nombre de versions écrites."""
        if fresh is None or len(fresh) == 0:
            return 0
        stamp = utc_stamp(fetched_at)
        incoming = _normalize(fresh)
        known = {_key(row): _signature(row) for row in self.as_known_at().to_dict("records")}
        rows = []
        for row in incoming.to_dict("records"):
            key = _key(row)
            signature = _signature(row)
            if known.get(key) == signature:
                continue
            known[key] = signature
            rows.append({**row, "fetched_at": stamp})
        if not rows:
            return 0
        out = pd.DataFrame(rows, columns=list(COLUMNS))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            out.to_csv(handle, header=header, index=False)
        return len(rows)

    def log_fetch(self, entry: Mapping[str, Any]) -> None:
        self.fetch_log.parent.mkdir(parents=True, exist_ok=True)
        with self.fetch_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(entry), ensure_ascii=False, default=str) + "\n")

    def fetches(self) -> list[dict]:
        if not self.fetch_log.exists():
            return []
        return [json.loads(line) for line in self.fetch_log.read_text(encoding="utf-8").splitlines() if line.strip()]
