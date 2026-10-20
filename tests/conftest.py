"""Garde-fou : aucun test n'écrit dans les vrais data/ et config/ du dépôt.

Le 20/10/2026, test_parse_oi_df écrivait deux faux open interest dans
data/cocoa_oi_receipts.csv, que l'entraînement lit. Ici, on prend une empreinte
(taille, date de modification) de chaque fichier avant et après la session :
une différence fait échouer la suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED_DIRS = ("data", "config")


def snapshot_guarded(root: Path = ROOT) -> dict[str, tuple[int, int]]:
    files: dict[str, tuple[int, int]] = {}
    for name in GUARDED_DIRS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def changed_files(before: dict, after: dict) -> list[str]:
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


def pytest_sessionstart(session):
    session.config._guarded_snapshot = snapshot_guarded()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_guarded_snapshot", None)
    if before is None:
        return
    changed = changed_files(before, snapshot_guarded())
    if not changed:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = "Des tests ont modifié des fichiers réels sous data/ ou config/ : " + ", ".join(changed[:20])
    if reporter is not None:
        reporter.write_line(message, red=True, bold=True)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
