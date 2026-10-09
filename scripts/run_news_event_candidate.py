"""Three NLP candidates. They are not fit while replayable history is too thin.

compose_served_price and the active cocoa release are not touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "audit_news_coverage", ROOT / "scripts" / "audit_news_coverage.py"
)
_audit = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_audit)
OUT = _audit.OUT
assess = _audit.assess
write_report = _audit.write_report


def main() -> int:
    summary = assess()
    write_report(summary)
    report = {
        "candidates": [
            "prophet_xgboost",
            "prophet_xgboost_sentiment",
            "prophet_xgboost_events",
        ],
        "horizons": [30, 60, 90],
        "history_sufficient": summary["history_sufficient"],
        "manifest_updated": False,
        "reason": (
            "Historique rejouable insuffisant. Aucun candidat n'est ajusté."
            if not summary["history_sufficient"]
            else "Historique suffisant. L'ajustement reste hors de cette commande tant que le manifeste n'est pas visé."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "candidate_gate.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report["reason"])
    print(f"[OK] {path}")
    print("[OK] Manifeste inchangé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
