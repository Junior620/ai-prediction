"""update_system.bat : enveloppe de scripts/run_pipeline.py, sans caractère invisible ni pause cachée."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "update_system.bat"
PIPELINE = ROOT / "scripts" / "run_pipeline.py"


def _text() -> str:
    return BAT.read_text(encoding="utf-8")


def test_no_control_character_in_the_batch():
    # Un « \b » mal échappé était devenu l'octet 0x08 : scripts<BS>uild_cocoa_roll_series.py.
    data = BAT.read_bytes()
    assert [b for b in data if b < 32 and b not in (9, 10, 13)] == []
    assert b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n")


def test_the_batch_only_wraps_the_pipeline():
    text = _text()
    called = re.findall(r"venv_py311\\Scripts\\python\.exe\s+([\w\\/.-]+\.py)", text)
    assert called == ["scripts\\run_pipeline.py"]
    assert "call venv_py311\\Scripts\\python.exe scripts\\run_pipeline.py %*" in text
    assert "docker compose" not in text


def test_every_script_named_by_the_pipeline_exists():
    source = PIPELINE.read_text(encoding="utf-8")
    named = set(re.findall(r'"((?:scripts/)?[\w]+\.py)"', source))
    assert "scripts/build_cocoa_roll_series.py" in named and "scripts/check_roll_series.py" in named
    missing = [name for name in named if not (ROOT / name).exists()]
    assert missing == []


def test_unattended_mode_never_pauses():
    lines = [line.strip() for line in _text().splitlines()]
    pauses = [line for line in lines if re.search(r"\bpause\b", line) and not line.upper().startswith("REM")]
    assert pauses == ["if %UNATTENDED% equ 0 pause"]
    assert 'for %%a in (%*) do if /i "%%~a"=="--unattended" set UNATTENDED=1' in lines


def test_the_batch_returns_the_pipeline_exit_code():
    lines = [line.strip() for line in _text().splitlines()]
    assert "set FINAL=%ERRORLEVEL%" in lines
    assert lines[-1] == "exit /b %FINAL%"
    assert re.findall(r"exit /b \S+", _text()) == ["exit /b %FINAL%"]
    for code in ("0", "1", "2", "3", "4"):
        assert f"REM   {code} " in _text()
