"""Coverage of replayable cocoa events. Training must read this report first."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.event_features import redundant_pairs, feature_frame
from src.nlp.event_rules import parse_utc
from src.nlp.event_store import LOCAL_LOG, load_local

OUT = ROOT / "reports" / "news_events"
MIN_REPLAYABLE_ORIGINS = 40
HORIZONS = (30, 60, 90)


def assess(versions: list | None = None) -> dict:
    rows = versions if versions is not None else load_local()
    replayable = [row for row in rows if parse_utc(row.get("available_at")) is not None]
    verified = [row for row in rows if row.get("published_at_verified")]
    months = Counter()
    for row in replayable:
        stamp = parse_utc(row["available_at"])
        if stamp is not None:
            months[stamp.strftime("%Y-%m")] += 1
    distinct = len({row.get("event_key") or row.get("url") for row in replayable})
    origins = sorted({parse_utc(row["available_at"]).date().isoformat() for row in replayable if parse_utc(row.get("available_at"))})
    usable = {horizon: max(0, len(origins) - horizon) for horizon in HORIZONS}
    enough = all(count >= MIN_REPLAYABLE_ORIGINS for count in usable.values())
    frame = feature_frame(replayable, [parse_utc(day) for day in origins[-30:]]) if origins else None
    redundant = redundant_pairs(frame) if frame is not None and not frame.empty else []
    return {
        "articles": len(rows),
        "distinct_events": distinct,
        "replayable": len(replayable),
        "published_verified": len(verified),
        "published_verified_share": (len(verified) / len(rows)) if rows else 0.0,
        "replayable_share": (len(replayable) / len(rows)) if rows else 0.0,
        "months": dict(sorted(months.items())),
        "empty_note": "Les mois absents de months n'ont aucun événement rejouable.",
        "origins": len(origins),
        "usable_by_horizon": usable,
        "min_replayable_origins": MIN_REPLAYABLE_ORIGINS,
        "history_sufficient": bool(enough and distinct > 0),
        "redundant_pairs": [
            {"left": left, "right": right, "correlation": round(value, 3)}
            for left, right, value in redundant[:30]
        ],
        "source": str(LOCAL_LOG),
    }


def write_report(summary: dict | None = None) -> Path:
    payload = summary or assess()
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "coverage.json"
    md_path = OUT / "coverage.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Couverture des événements cacao",
        "",
        f"Articles versionnés : {payload['articles']}",
        f"Événements distincts rejouables : {payload['distinct_events']}",
        f"Part rejouable : {payload['replayable_share']:.1%}",
        f"Part avec date éditeur vérifiée : {payload['published_verified_share']:.1%}",
        f"Origines : {payload['origins']}",
        f"Historique suffisant pour entraîner : {payload['history_sufficient']}",
        "",
        "Origines utilisables par horizon :",
    ]
    for horizon, count in payload["usable_by_horizon"].items():
        lines.append(f"- J+{horizon} : {count}")
    lines.append("")
    lines.append("Mois couverts :")
    if payload["months"]:
        for month, count in payload["months"].items():
            lines.append(f"- {month} : {count}")
    else:
        lines.append("- aucun")
    lines.append("")
    lines.append(payload["empty_note"])
    if payload["redundant_pairs"]:
        lines.append("")
        lines.append("Paires trop corrélées, à laisser hors du candidat :")
        for pair in payload["redundant_pairs"]:
            lines.append(f"- {pair['left']} / {pair['right']} ({pair['correlation']})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    path = write_report()
    summary = json.loads((OUT / "coverage.json").read_text(encoding="utf-8"))
    print(f"[OK] {path}")
    print(f"Historique suffisant : {summary['history_sufficient']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
