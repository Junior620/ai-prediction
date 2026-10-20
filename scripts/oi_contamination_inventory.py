"""Inventaire des lots, rapports et lignes de journal touchés par le faux open interest.

Depuis le commit 0f4b415 (2026-10-09), ``apply_oi_receipts`` remplace l'open
interest du cacao par le registre ``data/cocoa_oi_receipts.csv``. Ce registre ne
contenait que les deux valeurs factices d'un test (voir
``scripts/quarantine_test_oi_rows.py``). Un lot entraîné depuis cette date qui
découpe ses arbres sur une variable d'open interest a donc appris de ces valeurs.

On mesure dans chaque modèle XGBoost cacao la part des découpages (poids) et du
gain portée par les variables d'open interest. Rien n'est modifié, sauf le
fichier d'inventaire écrit dans ``data/quarantine/oi_contamination.json``.

Usage:
    python scripts/oi_contamination_inventory.py [--root E:\\prediction]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OI_FEATURES = ("open_interest", "oi_change_1d", "oi_change_7d")
LEDGER_SINCE = "2026-10-09"  # commit 0f4b415 : le registre remplace l'open interest du cacao
STAMP = re.compile(r"(\d{8}_\d{6})")


def oi_usage(model) -> dict:
    """Part des découpages et du gain sur les variables d'open interest."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    weight = booster.get_score(importance_type="weight")
    gain = booster.get_score(importance_type="total_gain")
    total_w = float(sum(weight.values())) or 1.0
    total_g = float(sum(gain.values())) or 1.0
    used = {name: int(weight.get(name, 0)) for name in OI_FEATURES if weight.get(name)}
    return {
        "oi_splits": int(sum(used.values())),
        "oi_split_share": round(sum(used.values()) / total_w, 4),
        "oi_gain_share": round(sum(float(gain.get(name, 0.0)) for name in OI_FEATURES) / total_g, 4),
        "oi_features_used": sorted(used),
        "has_oi_features": any(name in (booster.feature_names or []) for name in OI_FEATURES),
    }


def _load(path: Path):
    import joblib

    return joblib.load(path)


def _lot_files(root: Path) -> dict[str, list[Path]]:
    lots: dict[str, list[Path]] = {}
    for path in list((root / "models" / "candidates" / "cocoa").glob("*/xgboost_*.pkl")) + list(
        (root / "models").glob("xgboost_*.pkl")
    ):
        match = STAMP.search(path.name)
        if match:
            lots.setdefault(match.group(1), []).append(path)
    return lots


def _trained_on(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d")


def inventory(root: Path) -> dict:
    lots = []
    for stamp, paths in sorted(_lot_files(root).items()):
        per_model = []
        for path in sorted(paths):
            try:
                usage = oi_usage(_load(path))
            except Exception as exc:  # modèle illisible : signalé, pas ignoré
                usage = {"error": str(exc)[:120]}
            per_model.append({"file": path.relative_to(root).as_posix(), **usage})
        readable = [m for m in per_model if "error" not in m]
        uses_oi = [m for m in readable if m.get("oi_splits", 0) > 0]
        trained_on = _trained_on(stamp)
        after_ledger = trained_on >= LEDGER_SINCE
        if not readable:
            verdict = "illisible"
        elif after_ledger and uses_oi:
            verdict = "contaminé"
        elif after_ledger:
            verdict = "non touché (n'utilise pas l'open interest)"
        else:
            verdict = "antérieur au registre (open interest Supabase, pas les valeurs factices)"
        lots.append(
            {
                "version": f"improved_{stamp}",
                "trained_on": trained_on,
                "models": len(per_model),
                "models_using_oi": len(uses_oi),
                "max_oi_gain_share": max((m.get("oi_gain_share", 0.0) for m in readable), default=0.0),
                "mean_oi_gain_share": round(
                    sum(m.get("oi_gain_share", 0.0) for m in readable) / len(readable), 4
                )
                if readable
                else None,
                "verdict": verdict,
                "detail": per_model,
            }
        )
    contaminated = {lot["version"] for lot in lots if lot["verdict"] == "contaminé"}
    journal_counts = Counter()
    journal = root / "data" / "prediction_journal.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            version = str(row.get("model_version") or "")
            if version in contaminated:
                journal_counts[version] += 1
    reports = []
    for pattern in (
        "reports/walk_forward/daily/*_summary.json",
        "reports/external_signals/*",
        "reports/revision_stability/cocoa/causal_lot/*.json",
    ):
        for path in sorted(root.glob(pattern)):
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
            if stamp >= LEDGER_SINCE:
                reports.append({"file": path.relative_to(root).as_posix(), "modified_on": stamp})
    return {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger_since": LEDGER_SINCE,
        "cause": "tests/test_databento_london_collector.py::test_parse_oi_df écrivait dans data/cocoa_oi_receipts.csv",
        "contaminated_versions": sorted(contaminated),
        "journal_rows_by_version": dict(sorted(journal_counts.items())),
        "reports_after_ledger": reports,
        "not_affected": [
            "grille J+30 (variante sans oi_change_*, protocole du 20/10)",
            "fourchette de risque (aucune variable)",
            "prix affichés depuis le 13/10 (cours inchangé)",
            "robusta (apply_oi_receipts ne s'applique qu'au cacao)",
        ],
        "lots": lots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    report = inventory(root)
    output = Path(args.output) if args.output else root / "data" / "quarantine" / "oi_contamination.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for lot in report["lots"]:
        print(
            f"{lot['version']} ({lot['trained_on']}) : {lot['models_using_oi']}/{lot['models']} modèles "
            f"utilisent l'open interest, gain max {lot['max_oi_gain_share']:.1%} -> {lot['verdict']}"
        )
    print(f"Lignes de journal des lots contaminés : {sum(report['journal_rows_by_version'].values())}")
    print(f"Rapports produits depuis le {LEDGER_SINCE} : {len(report['reports_after_ledger'])}")
    print(f"[OK] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
