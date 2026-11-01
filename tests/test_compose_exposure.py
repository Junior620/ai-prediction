"""Redis et MLflow ne sont jamais publiés sur Internet ; le vidage du cache suit le mot de passe."""

from pathlib import Path

import yaml

from scripts import run_pipeline as rp

ROOT = Path(__file__).resolve().parents[1]


def _services():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_redis_and_mlflow_listen_on_localhost_only():
    # Docker contourne ufw : un port publié sans adresse est ouvert à tout Internet.
    # Redis a été attaqué ainsi du 29 au 31/10/2026, MLflow sondé le 30/10.
    services = _services()
    for name, port in (("redis", "6379"), ("mlflow", "5000"), ("api", "8000")):
        ports = services[name].get("ports") or []
        assert ports, name
        assert all(str(p).startswith("127.0.0.1:") and str(p).endswith(f":{port}") for p in ports), (name, ports)


def test_redis_uses_the_password_from_env_when_set():
    redis = _services()["redis"]
    assert "REDIS_PASSWORD=${REDIS_PASSWORD:-}" in redis["environment"]
    command = " ".join(redis["command"])
    assert '${REDIS_PASSWORD:+--requirepass "$$REDIS_PASSWORD"}' in command.replace("$${", "${")
    assert "PONG" in " ".join(redis["healthcheck"]["test"])
    # Le fichier de configuration monté ne peut pas être réécrit depuis Redis (CONFIG REWRITE).
    assert any(v.endswith("redis.conf:ro") for v in redis["volumes"])


def test_cache_flush_reads_the_redis_password():
    assert "password=os.getenv('REDIS_PASSWORD') or None" in rp.REDIS_FLUSH
