"""update_system.bat : aucun caractère invisible, scripts présents, codes de sortie distincts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "update_system.bat"


def _text() -> str:
    return BAT.read_text(encoding="utf-8")


def test_no_control_character_in_the_batch():
    # Un « \b » mal échappé était devenu l'octet 0x08 : scripts<BS>uild_cocoa_roll_series.py.
    data = BAT.read_bytes()
    assert [b for b in data if b < 32 and b not in (9, 10, 13)] == []
    assert b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n")


def test_every_local_python_script_exists():
    text = _text()
    called = set(re.findall(r"venv_py311\\Scripts\\python\.exe\s+([\w\\/.-]+\.py)", text))
    assert "scripts\\build_cocoa_roll_series.py" in called
    assert "scripts\\check_roll_series.py" in called
    missing = [name for name in called if not (ROOT / name.replace("\\", "/")).exists()]
    assert missing == []


def test_unattended_mode_never_pauses():
    lines = [line.strip() for line in _text().splitlines()]
    pauses = [line for line in lines if re.search(r"\bpause\b", line) and not line.upper().startswith("REM")]
    assert pauses == ["if !UNATTENDED! equ 0 pause"]
    assert 'if /i "%~1"=="--unattended" set UNATTENDED=1' in lines


def test_exit_codes_are_distinct():
    text = _text()
    for code in ("0", "1", "2", "3", "4"):
        assert f"if !FINAL! equ {code} echo" in text
    assert "exit /b !FINAL!" in text
    assert "set FINAL=2" in text and "set FINAL=3" in text and "set FINAL=4" in text
    # Aucune sortie directe qui court-circuiterait le code final.
    assert re.findall(r"exit /b (\d)", text) == ["0"]  # seulement la sous-routine wait_for_api


def test_roll_series_check_blocks_training():
    text = _text()
    check = text.index("scripts\\check_roll_series.py")
    train = text.index("train_hybrid_improved.py --market cocoa")
    assert check < train


def test_health_503_is_not_ready():
    block = _text().split("\n:wait_for_api", 1)[1].split("\n:finish", 1)[0]
    code = [line for line in block.splitlines() if not line.strip().upper().startswith("REM")]
    assert "Invoke-WebRequest" in "\n".join(code)
    assert "503" not in "\n".join(code)
