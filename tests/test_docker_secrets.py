"""Les secrets ne sont ni dans l'image Docker ni dans git."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_never_copies_an_env_file():
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    instructions = [line.strip() for line in lines if line.strip().upper().startswith(("COPY", "ADD"))]
    assert instructions, "le Dockerfile doit copier le code"
    assert not [line for line in instructions if re.search(r"\.env\b", line)]
    # Copier tout le contexte (« COPY . . ») ramènerait .env si .dockerignore l'oubliait.
    assert not [line for line in instructions if re.match(r"(COPY|ADD)\s+\.\s", line, re.I)]


def test_dockerignore_excludes_secrets_and_data():
    ignored = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    for pattern in (".env", ".env.*", "**/.env", "data/", "models/", ".git/"):
        assert pattern in ignored


def test_compose_provides_secrets_at_runtime_only():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "env_file:" in compose
    assert "./.env:/app/.env:ro" in compose


def test_no_env_file_is_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    secrets = [name for name in tracked if re.search(r"(^|/)\.env($|\.)", name) and not name.endswith(".example")]
    assert secrets == []
