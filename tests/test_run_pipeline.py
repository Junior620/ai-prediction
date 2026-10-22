"""Orchestrateur : étapes, codes de sortie, garde d'entraînement, production jamais touchée."""

import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_pipeline as rp
from scripts.freeze_market_snapshot import cut_incomplete
from src.models.revision_command import preview_served

ROOT = Path(__file__).resolve().parents[1]
TUESDAY = datetime(2026, 10, 20, 6, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 10, 24, 6, 0, tzinfo=timezone.utc)
IN_PLACE = {"cocoa": "20261019_052744", "coffee_robusta": "20261019_053000"}
NEW = {"cocoa": "20261024_061500", "coffee_robusta": "20261024_062000"}
ACTIVE = {"cocoa": "improved_20261006_213603", "coffee_robusta": "improved_20261006_220101"}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """Un dépôt minimal : lots en place, instantanés, fraîcheur, série sans saut, journal vide."""
    for market in rp.MARKETS:
        _write(tmp_path / "config" / f"experimental_release_{market}.json", {"version": f"improved_{IN_PLACE[market]}"})
        _write(tmp_path / "config" / f"active_release_{market}.json", {"market": market, "version": ACTIVE[market]})
        _write(tmp_path / "models" / "candidates" / market / "TRAINED_LOT.json", {"stamp": IN_PLACE[market]})
        _write(tmp_path / "data" / "status" / f"freshness_{market}.json", {"status": "ok", "expected_last_session": "2026-10-19"})
        snapshot = tmp_path / rp.SNAPSHOT[market]
        _write(snapshot, "date,price\n2026-10-16,4376.0\n2026-10-19,4377.0\n")
        _write(snapshot.with_suffix(".json"), {"max_date": "2026-10-19", "raw_max_date": "2026-10-19"})
    days = pd.bdate_range("2025-01-01", "2026-10-19")
    roll = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "price": 4000.0})
    roll.to_csv(tmp_path / "data" / "cocoa_roll_series.csv", index=False)
    _write(tmp_path / "data" / "prediction_journal.jsonl", "")
    return tmp_path


class FakeRunner:
    """Rejoue des codes de sortie ; un entraînement réussi nomme un nouveau lot."""

    def __init__(self, root: Path, codes=None):
        self.root = root
        self.codes = codes or {}
        self.calls = []

    def __call__(self, label, cmd, *, env=None):
        self.calls.append((label, list(cmd), dict(env or {})))
        text = " ".join(cmd)
        code = 0
        for needle, value in self.codes.items():
            if needle in text:
                code = value.pop(0) if isinstance(value, list) else value
                break
        if code == 0 and "train_hybrid_improved.py" in text:
            market = cmd[cmd.index("--market") + 1]
            _write(self.root / "models" / "candidates" / market / "TRAINED_LOT.json", {"stamp": NEW[market]})
        return code

    def commands(self, needle=""):
        return [" ".join(cmd) for _, cmd, _ in self.calls if needle in " ".join(cmd)]

    def index(self, needle):
        for i, (_, cmd, _) in enumerate(self.calls):
            if needle in " ".join(cmd):
                return i
        raise AssertionError(f"{needle} jamais lancé")


def _pipeline(repo, runner, *, now=TUESDAY, probe=lambda: 200, public_health=None, alert=None, **flags):
    args = Namespace(
        mode=flags.get("mode", "daily"),
        retrain=flags.get("retrain", False),
        dry_run=flags.get("dry_run", False),
        preview_journal=flags.get("preview_journal", False),
        unattended=True,
        markets=flags.get("markets", list(rp.MARKETS)),
        host=flags.get("host", "pc"),
    )
    pipeline = rp.Pipeline(
        args, runner, now=now, sleep=lambda _: None, probe=probe,
        public_health=public_health or (lambda: _health()), alert=alert, root=repo,
    )
    pipeline._api_token = lambda: "test-token"
    return pipeline


def _health(mode="active", versions=None, status="ok", last="2026-10-19", roll="2026-10-19"):
    """/health du VPS : mode, lot servi par marché, fraîcheur du cacao."""
    versions = versions or ACTIVE
    serving = {market: {"release_mode": mode, "model_version": versions[market]} for market in rp.MARKETS}
    serving["cocoa"]["freshness"] = {"status": status, "last_session": last, "roll_series_end": roll}
    return {"release_mode": mode, "status": "healthy", "serving": serving}


def _writer(repo, host):
    _write(repo / "config" / "config.yaml", f"pipeline:\n  journal_writer: {host}\n")


def test_an_ordinary_day_recomputes_forecasts_with_the_lot_in_place(repo):
    runner = FakeRunner(repo)
    code = _pipeline(repo, runner).execute()
    assert code == rp.EXIT_OK
    assert runner.commands("train_hybrid_improved.py") == []
    journal = runner.commands("journal_forecast_revisions.py")
    assert len(journal) == 2 and all("--preview" not in cmd for cmd in journal)
    assert f"--stamp {IN_PLACE['cocoa']}" in journal[0] and f"--stamp {IN_PLACE['coffee_robusta']}" in journal[1]
    ups = runner.commands("compose up -d")
    assert ups and all("--force-recreate" not in cmd for cmd in ups)
    assert runner.commands("deploy_models.ps1")[0].endswith("-DataOnly")
    assert runner.commands("run_walk_forward_validation.py") == []
    assert runner.commands("evaluate_predictions.py")
    decision = json.loads((repo / "data" / "status" / "retrain_cocoa.json").read_text(encoding="utf-8"))
    assert decision["retrain"] is False


def test_the_freeze_stops_at_the_last_complete_day(repo):
    runner = FakeRunner(repo)
    _write(repo / "logs" / f"databento_collection_{TUESDAY.astimezone():%Y%m%d}.json",
           [{"attempts": [{"strategy": "databento_ohlcv", "last_complete_day": "2026-10-19"}]}])
    _pipeline(repo, runner).execute()
    freeze = runner.commands("freeze_market_snapshot.py")
    assert "--market cocoa" in freeze[0] and freeze[0].endswith("--until 2026-10-19")
    # Robusta : règle d'horloge, 06:00 UTC le mardi -> lundi.
    assert freeze[1].endswith("--until 2026-10-19")


def test_saturday_trains_then_reloads_and_deploys_in_full(repo):
    runner = FakeRunner(repo)
    code = _pipeline(repo, runner, now=SATURDAY).execute()
    assert code == rp.EXIT_OK
    train = runner.commands("train_hybrid_improved.py")
    assert "--calendar trading_sessions" in train[0] and "--calendar keep_closed_rows" in train[1]
    assert runner.index("check_roll_series.py") < runner.index("train_hybrid_improved.py --market cocoa")
    journal = runner.commands("journal_forecast_revisions.py")
    assert f"--stamp {NEW['cocoa']}" in journal[0]
    assert runner.commands("write_experimental_release.py --market cocoa")[0].endswith(f"--stamp {NEW['cocoa']}")
    assert any("--force-recreate" in cmd for cmd in runner.commands("compose up -d"))
    assert not runner.commands("deploy_models.ps1")[0].endswith("-DataOnly")
    walk = runner.commands("run_walk_forward_validation.py")
    assert len(walk) == 2 and all("--skip-calibration" in cmd and "--measure-only" in cmd for cmd in walk)
    assert runner.commands("train_futures_curve.py") and runner.commands("cleanup_old_models.py")


def test_an_incomplete_session_blocks_training_and_exits_2(repo):
    stamp = repo / rp.SNAPSHOT["cocoa"]
    _write(stamp.with_suffix(".json"), {"max_date": "2026-10-23", "raw_max_date": "2026-10-24"})
    _write(repo / "data" / "status" / "freshness_cocoa.json", {"status": "ok", "expected_last_session": "2026-10-23"})
    runner = FakeRunner(repo)
    code = _pipeline(repo, runner, now=SATURDAY, markets=["cocoa"]).execute()
    assert code == rp.EXIT_STALE
    assert runner.commands("train_hybrid_improved.py") == []
    assert f"--stamp {IN_PLACE['cocoa']}" in runner.commands("journal_forecast_revisions.py")[0]
    decision = json.loads((repo / "data" / "status" / "retrain_cocoa.json").read_text(encoding="utf-8"))
    assert decision["wanted"] and not decision["retrain"] and "incomplète" in decision["blocked_by"][0]


def test_an_incomplete_session_is_reported_even_without_a_retrain(repo):
    stamp = repo / rp.SNAPSHOT["cocoa"]
    _write(stamp.with_suffix(".json"), {"max_date": "2026-10-19", "raw_max_date": "2026-10-20"})
    runner = FakeRunner(repo)
    assert _pipeline(repo, runner, markets=["cocoa"]).execute() == rp.EXIT_STALE
    assert runner.commands("train_hybrid_improved.py") == []
    assert runner.commands("journal_forecast_revisions.py")


def test_a_missing_session_blocks_training(repo):
    _write(repo / "data" / "status" / "freshness_cocoa.json", {"status": "stale", "reason": "retard de 1 séance(s)"})
    runner = FakeRunner(repo, {"collect_ice_london_cocoa.py": 2})
    code = _pipeline(repo, runner, now=SATURDAY, markets=["cocoa"]).execute()
    assert code == rp.EXIT_STALE
    assert runner.commands("train_hybrid_improved.py") == []
    assert runner.commands("journal_forecast_revisions.py")


def test_a_roll_series_behind_the_snapshot_stops_everything(repo):
    runner = FakeRunner(repo, {"check_roll_series.py": 1})
    code = _pipeline(repo, runner, now=SATURDAY).execute()
    assert code == rp.EXIT_FAILED
    assert runner.commands("train_hybrid_improved.py") == []
    assert runner.commands("journal_forecast_revisions.py") == []
    assert runner.commands("collect_coffee_robusta_price.py") == []


def test_a_failed_training_keeps_the_lot_in_place_and_still_journals(repo):
    runner = FakeRunner(repo, {"train_hybrid_improved.py --market cocoa": 1})
    code = _pipeline(repo, runner, now=SATURDAY, markets=["cocoa"]).execute()
    assert code == rp.EXIT_FAILED
    assert f"--stamp {IN_PLACE['cocoa']}" in runner.commands("journal_forecast_revisions.py")[0]
    assert all("--force-recreate" not in cmd for cmd in runner.commands("compose up -d"))


def test_robusta_quarantine_and_retries_never_remove_cocoa(repo):
    runner = FakeRunner(repo, {"collect_coffee_robusta_price.py": 4})
    assert _pipeline(repo, runner).execute() == rp.EXIT_QUARANTINE
    assert len(runner.commands("collect_coffee_robusta_price.py")) == 1
    assert len(runner.commands("journal_forecast_revisions.py")) == 1
    runner = FakeRunner(repo, {"collect_coffee_robusta_price.py": [2, 2, 2]})
    assert _pipeline(repo, runner).execute() == rp.EXIT_STALE
    assert len(runner.commands("collect_coffee_robusta_price.py")) == 3
    runner = FakeRunner(repo, {"collect_coffee_robusta_price.py": [2, 0]})
    assert _pipeline(repo, runner).execute() == rp.EXIT_OK
    assert len(runner.commands("journal_forecast_revisions.py")) == 2


def test_health_503_is_not_ready(repo):
    runner = FakeRunner(repo)
    answers = iter([None, 503, 200] + [200] * 100)
    assert _pipeline(repo, runner, probe=lambda: next(answers), markets=["cocoa"]).execute() == rp.EXIT_OK
    runner = FakeRunner(repo)
    code = _pipeline(repo, runner, probe=lambda: 503, markets=["cocoa"]).execute()
    assert code == rp.EXIT_PARTIAL
    assert runner.commands("verify_served_forecast.py") == []


def test_exit_codes_keep_their_priority():
    assert rp.Outcome().code() == 0
    assert rp.Outcome(partial=["x"], quarantine=["y"], stale=["z"], failed=["w"]).code() == 1
    assert rp.Outcome(partial=["x"], quarantine=["y"], stale=["z"]).code() == 2
    assert rp.Outcome(partial=["x"], quarantine=["y"]).code() == 3
    assert rp.Outcome(partial=["x"]).code() == 4


def test_production_manifests_are_never_written(repo):
    active = {p: p.read_bytes() for p in (repo / "config").glob("active_release_*.json")}
    real = {p: p.stat().st_mtime_ns for p in (ROOT / "config").glob("active_release_*.json")}
    runner = FakeRunner(repo)
    _pipeline(repo, runner, now=SATURDAY, retrain=True).execute()
    assert {p: p.read_bytes() for p in (repo / "config").glob("active_release_*.json")} == active
    assert {p: p.stat().st_mtime_ns for p in (ROOT / "config").glob("active_release_*.json")} == real
    for _, cmd, _ in runner.calls:
        text = " ".join(cmd)
        assert "active_release" not in text and "promote" not in text
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "active_release_" not in source.replace("``config/active_release_*.json``", "")


def test_dry_run_writes_nothing_and_lists_both_paths(repo, capsys):
    before = sorted(p.relative_to(repo) for p in repo.rglob("*"))
    daily = rp.Runner(dry_run=True)
    assert _pipeline(repo, daily, dry_run=True).execute() == rp.EXIT_OK
    weekly = rp.Runner(dry_run=True)
    assert _pipeline(repo, weekly, dry_run=True, mode="weekly").execute() == rp.EXIT_OK
    assert sorted(p.relative_to(repo) for p in repo.rglob("*")) == before
    weekly_text = [" ".join(cmd) for _, cmd in weekly.calls]
    assert any("train_hybrid_improved.py" in cmd for cmd in weekly_text)
    assert not any("train_hybrid_improved.py" in " ".join(cmd) for _, cmd in daily.calls)
    assert "[ÉTAPE]" in capsys.readouterr().out


def test_preview_journal_only_computes(repo):
    runner = FakeRunner(repo)
    assert _pipeline(repo, runner, preview_journal=True).execute() == rp.EXIT_OK
    texts = runner.commands()
    assert len(texts) == 2 and all("--preview" in cmd for cmd in texts)
    assert texts[1].startswith("docker compose run")
    assert runner.calls[1][2] == rp.DOCKER_ENV


def test_cut_incomplete_drops_only_later_sessions():
    frame = pd.DataFrame({"date": ["2026-10-16", "2026-10-19", "2026-10-20"], "price": [1.0, 2.0, 3.0]})
    kept, raw_max, dropped = cut_incomplete(frame, "2026-10-19")
    assert list(kept["date"]) == ["2026-10-16", "2026-10-19"]
    assert raw_max == "2026-10-20" and dropped == ["2026-10-20"]
    same, _, none = cut_incomplete(frame, "")
    assert len(same) == 3 and none == []


def test_preview_rows_keep_naive_and_model_apart():
    def item(h, served_as_close, candidate):
        parts = {
            "display_final": True,
            "served_as_close": served_as_close,
            "candidate_price": candidate,
            "origin_price": 4377.0,
            "origin_date": "2026-10-19",
            "target_date": "2026-10-20",
        }
        return SimpleNamespace(horizon=h, status="experimental", price=4377.0, components=parts, confidence_interval=[])

    rows = preview_served([item(1, True, 4390.0), item(90, True, 4500.0)], release={"version": "improved_x"})
    assert rows[0]["naive_price"] == 4377.0 and rows[0]["served_kind"] == "unchanged_close"
    assert rows[0]["candidate_kind"] == "experimental_model" and rows[0]["candidate_price"] == 4390.0
    assert rows[1]["journal_only"] is True and rows[0]["journal_only"] is False


def test_data_only_deploy_sends_series_without_restart():
    deploy = (ROOT / "deploy_models.ps1").read_text(encoding="utf-8")
    assert "[switch]$DataOnly" in deploy
    start = deploy.index("if ($DataOnly) {")
    block = deploy[start:deploy.index('$cocoaRelease = Get-ActiveRelease "cocoa"')]
    assert "_roll_series.csv" in block and "freshness" in block
    assert "docker compose restart" not in block and "Send-RepoFile" not in block
    # Le bloc sort avant tout envoi de modèle, de code ou de manifeste.
    assert start < deploy.index("Send-RepoFile $cocoaRelease.artifacts.prophet")


# ---------------------------------------------------------------------- lot C : VPS
VPS_SCRIPTS = ROOT / "scripts" / "vps"


def _no_docker_and_no_experimental_mode(runner):
    for _, cmd, env in runner.calls:
        text = " ".join(cmd)
        assert not cmd[0].startswith("docker") and "compose" not in text, text
        assert "deploy_models.ps1" not in text and "powershell" not in text, text
        assert "PREDICTION_RELEASE_MODE" not in env, text


def test_vps_runs_every_step_in_its_container_and_leaves_the_public_api_alone(repo):
    _writer(repo, "vps")
    runner = FakeRunner(repo)
    asked = []
    code = _pipeline(repo, runner, host="vps", public_health=lambda: asked.append(1) or _health()).execute()
    assert code == rp.EXIT_OK
    _no_docker_and_no_experimental_mode(runner)
    journal = runner.calls[runner.index("journal_forecast_revisions.py --market coffee_robusta")][1]
    assert journal[0] == sys.executable and journal[1] == "scripts/journal_forecast_revisions.py"
    assert runner.commands("verify_served_forecast.py") == []
    assert runner.commands("flushdb()") and asked
    assert runner.index("evaluate_predictions.py") < runner.index("flushdb()") < runner.index("src.monitoring.freshness")
    assert len(runner.commands("write_experimental_release.py")) == 2


def test_vps_saturday_trains_in_its_container_and_never_writes_production_manifests(repo):
    _writer(repo, "vps")
    active = {p: p.read_bytes() for p in (repo / "config").glob("active_release_*.json")}
    real = {p: p.stat().st_mtime_ns for p in (ROOT / "config").glob("active_release_*.json")}
    runner = FakeRunner(repo)
    code = _pipeline(repo, runner, now=SATURDAY, host="vps").execute()
    assert code == rp.EXIT_OK
    _no_docker_and_no_experimental_mode(runner)
    assert len(runner.commands("train_hybrid_improved.py")) == 2
    nhits = runner.calls[runner.index("train_nhits.py")][1]
    assert nhits[:3] == [sys.executable, "-u", "train_nhits.py"]
    assert runner.commands("write_experimental_release.py --market cocoa")[0].endswith(f"--stamp {NEW['cocoa']}")
    assert len(runner.commands("run_walk_forward_validation.py")) == 2
    assert {p: p.read_bytes() for p in (repo / "config").glob("active_release_*.json")} == active
    assert {p: p.stat().st_mtime_ns for p in (ROOT / "config").glob("active_release_*.json")} == real
    assert all("active_release" not in " ".join(cmd) for _, cmd, _ in runner.calls)


def test_only_the_journal_writer_runs_for_real(repo, capsys):
    runner = FakeRunner(repo)
    assert rp.journal_writer(repo) == "pc"  # pas de config.yaml : le PC, le VPS refuse
    assert _pipeline(repo, runner, host="vps").execute() == rp.EXIT_FAILED
    assert runner.calls == []
    _writer(repo, "vps")
    assert _pipeline(repo, runner, host="pc").execute() == rp.EXIT_FAILED
    assert runner.calls == []
    assert "pipeline.journal_writer" in capsys.readouterr().out
    # Le dry-run reste permis et le dit.
    dry = rp.Runner(dry_run=True)
    assert _pipeline(repo, dry, host="pc", dry_run=True).execute() == rp.EXIT_OK
    assert "serait refusé" in capsys.readouterr().out and dry.calls


def test_vps_dry_run_writes_nothing_and_lists_the_public_checks(repo, capsys):
    before = sorted(p.relative_to(repo) for p in repo.rglob("*"))
    for mode in ("daily", "weekly"):
        dry = rp.Runner(dry_run=True)
        pipeline = _pipeline(repo, dry, host="vps", dry_run=True, mode=mode, public_health=lambda: None)
        assert pipeline.execute() == rp.EXIT_OK
        assert not any(cmd[0].startswith("docker") or "deploy_models" in " ".join(cmd) for _, cmd in dry.calls)
    assert sorted(p.relative_to(repo) for p in repo.rglob("*")) == before
    out = capsys.readouterr().out
    assert "mode active, lot actif" in out and "cache Redis" in out


def test_public_api_outside_active_mode_is_a_failure(repo):
    _writer(repo, "vps")
    code = _pipeline(repo, FakeRunner(repo), host="vps", markets=["cocoa"],
                     public_health=lambda: _health(mode="experimental")).execute()
    assert code == rp.EXIT_FAILED
    served = {**ACTIVE, "cocoa": f"improved_{IN_PLACE['cocoa']}"}
    pipeline = _pipeline(repo, FakeRunner(repo), host="vps", markets=["cocoa"],
                         public_health=lambda: _health(versions=served))
    assert pipeline.execute() == rp.EXIT_FAILED
    assert "l'API publique sert" in pipeline.outcome.failed[0]


def test_public_api_late_or_unreachable_is_reported(repo):
    _writer(repo, "vps")
    answers = iter([None, _health(last="2026-10-16"), _health()])
    assert _pipeline(repo, FakeRunner(repo), host="vps", public_health=lambda: next(answers)).execute() == rp.EXIT_OK
    pipeline = _pipeline(repo, FakeRunner(repo), host="vps", public_health=lambda: None)
    assert pipeline.execute() == rp.EXIT_PARTIAL
    assert "injoignable" in pipeline.outcome.partial[0]
    # Conteneur sans ./data monté : la série sans saut n'arrive pas (VPS du 20/10/2026).
    pipeline = _pipeline(repo, FakeRunner(repo), host="vps", public_health=lambda: _health(roll=None))
    assert pipeline.execute() == rp.EXIT_PARTIAL
    assert "série sans saut" in pipeline.outcome.partial[0]


def test_alert_and_last_run_follow_the_exit_code(repo):
    _writer(repo, "vps")
    sent = []
    code = _pipeline(repo, FakeRunner(repo, {"collect_ice_london_cocoa.py": 2}), host="vps",
                     alert=lambda c, o: sent.append((c, list(o.stale)))).execute()
    assert code == rp.EXIT_STALE and sent and sent[0][0] == rp.EXIT_STALE
    last = json.loads((repo / rp.LAST_RUN).read_text(encoding="utf-8"))
    assert last["host"] == "vps" and last["code"] == rp.EXIT_STALE and last["stale"]
    sent.clear()
    assert _pipeline(repo, FakeRunner(repo), host="vps", alert=lambda c, o: sent.append(c)).execute() == rp.EXIT_OK
    assert sent == []

    def broken(code, outcome):
        raise RuntimeError("webhook absent")

    runner = FakeRunner(repo, {"check_roll_series.py": 1})
    assert _pipeline(repo, runner, host="vps", alert=broken).execute() == rp.EXIT_FAILED


def test_host_comes_from_the_flag_or_the_environment(monkeypatch):
    monkeypatch.delenv("PIPELINE_HOST", raising=False)
    assert rp.parse_args([]).host == "pc"
    monkeypatch.setenv("PIPELINE_HOST", "vps")
    assert rp.parse_args([]).host == "vps" and rp.parse_args(["--host", "pc"]).host == "pc"
    monkeypatch.setenv("PIPELINE_HOST", "serveur")
    with pytest.raises(SystemExit):
        rp.parse_args([])


def _code_lines(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and not line.lstrip().startswith("#")]


def test_vps_wrapper_never_recreates_or_switches_the_public_api():
    text = "\n".join(_code_lines(VPS_SCRIPTS / "pipeline.sh"))
    for forbidden in ("compose up", "restart", "compose build", "--build", "PREDICTION_RELEASE_MODE=experimental", "active_release"):
        assert forbidden not in text, forbidden
    assert "unset PREDICTION_RELEASE_MODE" in text and "grep -qx 'PREDICTION_RELEASE_MODE=active' .env" in text
    assert "docker compose run --rm --no-deps -T" in text and "--host vps" in text and "--alert" in text
    assert "flock -n" in text and "docker image inspect" in text
    # Le journal du traitement ne suit jamais celui de l'API publique (.env : PREDICTION_JOURNAL).
    assert "-e PREDICTION_JOURNAL= " in text
    assert b"\r" not in (VPS_SCRIPTS / "pipeline.sh").read_bytes()


def test_cron_runs_after_databento_publishes_in_summer_and_winter():
    from zoneinfo import ZoneInfo

    berlin = ZoneInfo("Europe/Berlin")
    entries = [line.split() for line in _code_lines(VPS_SCRIPTS / "prediction-pipeline.cron") if line[0].isdigit()]
    assert entries and all(e[5] == "root" and e[6] == "/opt/prediction/scripts/vps/pipeline.sh" for e in entries)
    main = [e for e in entries if len(e) == 7]
    assert len(main) == 1 and any(e[7:] == ["--catch-up"] for e in entries)
    minute, hour, _, _, days = main[0][:5]
    assert days == "0,2-6"  # lendemain local des séances lun-ven, et samedi soir UTC
    # Séance du lundi 19/01 (hiver) et du lundi 20/07 (été) : passage le mardi 00:30 à Berlin.
    for session in (datetime(2026, 1, 19), datetime(2026, 7, 20)):
        local = datetime(session.year, session.month, session.day + 1, int(hour), int(minute), tzinfo=berlin)
        utc = local.astimezone(timezone.utc)
        assert utc.date() == session.date() and (utc.hour, utc.minute) >= (22, 5)


def test_vps_update_keeps_pipeline_manifests_and_refuses_unknown_content():
    import re

    script = (VPS_SCRIPTS / "update.sh").read_text(encoding="utf-8")
    keep = re.compile(re.search(r"KEEP_RE='([^']+)'", script).group(1))
    assert keep.match("config/experimental_release_cocoa.json")
    assert keep.match("config/previous_experimental_release_coffee_robusta.json")
    assert not keep.match("config/active_release_cocoa.json") and not keep.match("config/config.yaml")
    code = "\n".join(_code_lines(VPS_SCRIPTS / "update.sh"))
    assert "exit 3" in code and "tar czf" in code and "git checkout -q -f -B main" in code
    # Suivis par HEAD, plus par la cible (88 modèles de bcab53e) : remis sur le disque s'ils sont ignorés.
    assert "--diff-filter=D HEAD" in code and "git check-ignore -q" in code
    assert code.index("--diff-filter=D HEAD") < code.index("git checkout -q -f -B main") < code.index("git check-ignore -q")
    # Ni build ni redémarrage : seulement affichés, à lancer après accord.
    assert all(line.lstrip().startswith("echo") for line in code.splitlines() if "docker compose" in line)
