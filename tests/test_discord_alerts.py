"""Alertes Discord : format natif lisible, et alerte des échecs de pipeline.sh sans exposer l'URL."""

from pathlib import Path
from unittest.mock import Mock, patch

from src.monitoring.alert_system import (
    DISCORD_MAX_CONTENT,
    AlertSeverity,
    AlertSystem,
    AlertType,
    is_discord_webhook,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SH = ROOT / "scripts" / "vps" / "pipeline.sh"
DISCORD = "https://discord.com/api/webhooks/123/abc"


def test_discord_urls_are_recognised():
    assert is_discord_webhook(DISCORD)
    assert is_discord_webhook("https://discordapp.com/api/webhooks/123/abc?wait=true")
    # L'adaptateur Slack de Discord garde le format Slack.
    assert not is_discord_webhook(DISCORD + "/slack")
    assert not is_discord_webhook("https://hooks.slack.com/services/T/B/X")
    assert not is_discord_webhook(None)


def _system(url):
    with patch("src.monitoring.alert_system.get_settings") as settings:
        settings.return_value.alert_webhook_url = url
        settings.return_value.alert_email_enabled = False
        return AlertSystem()


@patch("requests.post")
def test_discord_receives_the_message_and_its_details(mock_post):
    mock_post.return_value = Mock(status_code=204, raise_for_status=Mock())
    sent = _system(DISCORD).send_alert(
        AlertSeverity.CRITICAL,
        AlertType.SYSTEM_ERROR,
        "Traitement VPS : code 1",
        details={"failed": ["cocoa : collecte ICE London échouée"], "host": "vps"},
    )
    assert sent is True
    url, payload = mock_post.call_args[0][0], mock_post.call_args[1]["json"]
    assert url == DISCORD and "blocks" not in payload
    content = payload["content"]
    assert content.startswith("🚨 **CRITICAL** — system_error")
    assert "Traitement VPS : code 1" in content and "collecte ICE London échouée" in content
    assert payload["allowed_mentions"] == {"parse": []}


@patch("requests.post")
def test_slack_keeps_its_blocks(mock_post):
    mock_post.return_value = Mock(status_code=200, raise_for_status=Mock())
    _system("https://hooks.slack.com/services/T/B/X").send_alert(
        AlertSeverity.ERROR, AlertType.SYSTEM_ERROR, "code 2"
    )
    assert "blocks" in mock_post.call_args[1]["json"]


def test_discord_content_stays_under_its_limit():
    payload = _system(DISCORD)._format_discord_payload(
        {
            "severity": "ERROR",
            "alert_type": "system_error",
            "message": "x" * 5000,
            "details": {},
            "timestamp": "2026-10-27T23:31:00",
        }
    )
    assert len(payload["content"]) <= DISCORD_MAX_CONTENT


def _code_lines():
    lines = PIPELINE_SH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#"))


def test_pipeline_sh_posts_its_own_failures_without_printing_the_url():
    code = _code_lines()
    assert "grep -m1 '^ALERT_WEBHOOK_URL=' .env" in code
    assert "jq -nc --arg t" in code and "curl -fsS -m 10" in code
    # La fonction alert() appelle le webhook ; l'URL n'est jamais écrite dans le journal.
    alert_body = code.split("alert() {", 1)[1].split("}", 1)[0]
    assert "notify " in alert_body
    for line in code.splitlines():
        if "$url" in line:
            assert "echo" not in line and "logger" not in line, line
    assert "--test-alert" in code
    assert b"\r" not in PIPELINE_SH.read_bytes()
