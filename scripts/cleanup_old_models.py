"""
Keep only the N most recent model artefacts per family under models/.

Families:
  - prophet_improved_*.pkl / model_info_improved_*.json
  - xgboost_improved_*.pkl
  - nhits_* directories + model_info_nhits_*.json
  - futures/*.pkl + model_info_futures.json (keep latest only)
  - coffee_robusta/ mirrors the same patterns
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _sorted_by_mtime(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _protected_paths() -> set[Path]:
    protected: set[Path] = set()
    names = (
        "active_release_cocoa.json",
        "active_release_coffee_robusta.json",
        "experimental_release_cocoa.json",
        "experimental_release_coffee_robusta.json",
    )
    for name in names:
        path = ROOT / "config" / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for relative in (payload.get("artifacts") or {}).values():
            artifact = Path(str(relative))
            if not artifact.is_absolute():
                artifact = ROOT / artifact
            protected.add(artifact.resolve())
    return protected


def _prune_files(files: list[Path], keep: int, protected: set[Path] | None = None) -> int:
    kept_safe = protected or set()
    removed = 0
    for p in _sorted_by_mtime(files)[keep:]:
        if p.resolve() in kept_safe:
            print(f"  kept file   {p.relative_to(ROOT)}")
            continue
        p.unlink(missing_ok=True)
        removed += 1
        print(f"  removed file {p.relative_to(ROOT)}")
    return removed


def _prune_dirs(dirs: list[Path], keep: int, protected: set[Path] | None = None) -> int:
    kept_safe = protected or set()
    removed = 0
    for p in _sorted_by_mtime(dirs)[keep:]:
        if p.resolve() in kept_safe:
            print(f"  kept dir    {p.relative_to(ROOT)}")
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed += 1
        print(f"  removed dir  {p.relative_to(ROOT)}")
    return removed


def cleanup_dir(base: Path, keep: int, protected: set[Path] | None = None) -> int:
    if not base.is_dir():
        return 0
    print(f"Cleanup {base.relative_to(ROOT)} (keep={keep})")
    removed = 0

    removed += _prune_files(list(base.glob("prophet_improved_*.pkl")), keep, protected)
    removed += _prune_files(list(base.glob("xgboost_improved_*.pkl")), keep, protected)
    removed += _prune_files(list(base.glob("model_info_improved_*.json")), keep, protected)
    removed += _prune_files(list(base.glob("model_info_nhits_*.json")), keep, protected)
    removed += _prune_dirs(
        [d for d in base.glob("nhits_*") if d.is_dir()],
        keep,
        protected,
    )

    futures = base / "futures"
    if futures.is_dir():
        # Keep latest info json + the N newest pkls overall
        removed += _prune_files(list(futures.glob("*.pkl")), keep)
        infos = list(futures.glob("model_info*.json"))
        if len(infos) > 1:
            removed += _prune_files(infos, 1)

    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=5)
    args = parser.parse_args()

    protected = _protected_paths()
    total = cleanup_dir(MODELS, args.keep, protected)
    total += cleanup_dir(MODELS / "coffee_robusta", args.keep, protected)
    print(f"Done. Removed {total} artefact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
