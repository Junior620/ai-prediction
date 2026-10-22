"""Traitement quotidien et hebdomadaire : collecter, figer, décider, prévoir, servir.

Chaque jour, pour le cacao puis le robusta :
  1. collecte (prix, courbes, météo, news, événements) ;
  2. série sans saut, gel des données jusqu'à la dernière journée complète, contrôle de la série ;
  3. décision de réentraînement (src/models/retrain_policy.py), soumise à la garde : jamais
     sur une séance incomplète ni quand la séance attendue manque ;
  4. prévisions recalculées et journalisées avec le lot en place (ou le nouveau lot) ;
  5. manifeste expérimental (inchangé si le lot ne change pas) ;
  6. API : ``up -d``, recréée seulement si un nouveau lot est chargé, puis contrôle servi.
Ensuite : évaluation, walk-forward et courbes les jours d'entraînement, déploiement des
données seules (ou complet après un entraînement), fraîcheur.

Ce traitement n'écrit jamais ``config/active_release_*.json`` : un nouveau lot reste
expérimental jusqu'à sa validation, et la promotion reste une décision manuelle.

Usage :
    python scripts/run_pipeline.py [--mode daily|weekly] [--retrain] [--dry-run]
                                   [--preview-journal] [--unattended]
Codes de sortie : 0 succès, 1 échec réel, 2 source en retard ou séance incomplète,
3 quarantaine, 4 succès partiel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models.release_calendar import MARKET_CALENDAR  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_STALE = 2
EXIT_QUARANTINE = 3
EXIT_PARTIAL = 4

MARKETS = ("cocoa", "coffee_robusta")
LABEL = {"cocoa": "Cacao", "coffee_robusta": "Robusta"}
SNAPSHOT = {
    "cocoa": "data/freeze/cocoa_prices.csv",
    "coffee_robusta": "data/freeze/coffee_robusta_prices.csv",
}
WALK_FORWARD_DIR = {
    "cocoa": "reports/walk_forward/daily",
    "coffee_robusta": "reports/walk_forward/coffee_robusta/daily",
}
ROLL_SERIES = {
    "cocoa": "data/cocoa_roll_series.csv",
    "coffee_robusta": "data/robusta_roll_series.csv",
}
# Journal de collecte Databento : il donne la dernière journée complète publiée.
COLLECTION_JOURNAL = {
    "cocoa": "databento_collection",
    "coffee_robusta": "databento_robusta_collection",
}
PREVIEW_DIR = "reports/pipeline"
HEALTH_URL = "http://localhost:8000/health"
DOCKER_ENV = {"PREDICTION_RELEASE_MODE": "experimental"}
ROBUSTA_COLLECT_TRIES = 3
ROBUSTA_RETRY_SECONDS = 45


@dataclass
class Outcome:
    failed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    quarantine: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    new_lots: dict[str, str] = field(default_factory=dict)
    halted: bool = False

    def code(self) -> int:
        if self.failed:
            return EXIT_FAILED
        if self.stale:
            return EXIT_STALE
        if self.quarantine:
            return EXIT_QUARANTINE
        if self.partial:
            return EXIT_PARTIAL
        return EXIT_OK


class Runner:
    """Lance une commande depuis la racine du dépôt ; en --dry-run, l'affiche seulement."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, label: str, cmd: list[str], *, env: Optional[dict] = None) -> int:
        self.calls.append((label, list(cmd)))
        shown = " ".join(cmd)
        if self.dry_run:
            print(f"[ÉTAPE] {label}\n         {shown}")
            return 0
        print(f"\n[INFO] {label}\n       {shown}", flush=True)
        merged = dict(os.environ)
        merged.update(env or {})
        merged.setdefault("PYTHONIOENCODING", "utf-8")
        return subprocess.run(cmd, cwd=ROOT, env=merged).returncode


def health_status(url: str = HEALTH_URL) -> Optional[int]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def stamp_of(version: Optional[str]) -> str:
    text = str(version or "")
    return text.split("improved_", 1)[1] if text.startswith("improved_") else ""


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class Pipeline:
    def __init__(
        self,
        args: argparse.Namespace,
        runner: Callable[..., int],
        *,
        now: Optional[datetime] = None,
        sleep: Callable[[float], None] = time.sleep,
        probe: Callable[[], Optional[int]] = health_status,
        root: Path = ROOT,
    ):
        self.args = args
        self.run = runner
        self.now = now or datetime.now(timezone.utc)
        self.sleep = sleep
        self.probe = probe
        self.root = Path(root)
        self.py = sys.executable
        self.outcome = Outcome()
        self.decisions: dict = {}

    # ------------------------------------------------------------------ état sur disque
    def lot_in_place(self, market: str) -> Optional[str]:
        payload = _read_json(self.root / "config" / f"experimental_release_{market}.json") or {}
        return payload.get("version")

    def trained_stamp(self, market: str) -> str:
        payload = _read_json(self.root / "models" / "candidates" / market / "TRAINED_LOT.json") or {}
        return str(payload.get("stamp") or "")

    def last_complete_day(self, market: str) -> str:
        """Plage Databento lue par la collecte du marché ; sinon la règle d'horloge (21 h UTC)."""
        from src.models.retrain_policy import clock_last_complete_day

        # Même date locale que write_collection_journal.
        path = self.root / "logs" / f"{COLLECTION_JOURNAL[market]}_{self.now.astimezone():%Y%m%d}.json"
        entries = _read_json(path)
        if isinstance(entries, dict):
            entries = [entries]
        for entry in reversed(entries or []):
            for attempt in reversed(entry.get("attempts") or []):
                if attempt.get("last_complete_day"):
                    return str(attempt["last_complete_day"])
        return clock_last_complete_day(self.now)

    # ------------------------------------------------------------------ déroulé
    def execute(self) -> int:
        if self.args.preview_journal:
            return self.preview()
        if not self.args.dry_run and not self._api_token():
            print("[ERREUR] API_TOKEN absent. Ajoutez API_TOKEN=... dans .env (generate_jwt_token.py).")
            return EXIT_FAILED
        if self.run("Docker opérationnel", ["docker", "info"]) != 0:
            print("[ERREUR] Docker n'est pas en cours d'exécution.")
            return EXIT_FAILED
        for market in self.args.markets:
            self.market(market)
            if self.outcome.halted:
                return self.finish()
        self.after_markets()
        return self.finish()

    def market(self, market: str) -> None:
        label = LABEL[market]
        print(f"\n{'=' * 80}\n{label.upper()}\n{'=' * 80}")
        if not self.collect(market):
            return
        until = self.last_complete_day(market)
        snapshot = SNAPSHOT[market]
        freeze = [self.py, "scripts/freeze_market_snapshot.py", "--market", market, "--output", snapshot, "--until", until]
        if self.run(f"{label} : gel des données jusqu'au {until}", freeze) != 0:
            self._fail(market, "instantané non figé")
            return
        check = [self.py, "scripts/check_roll_series.py", "--market", market, "--snapshot", snapshot]
        if self.run(f"{label} : la série sans saut couvre l'instantané", check) != 0:
            self._fail(market, "série sans saut en retard sur l'instantané, rien n'est entraîné")
            return
        decision = self.decide(market, until)
        in_place = stamp_of(self.lot_in_place(market))
        stamp = in_place
        if decision.retrain:
            before = self.trained_stamp(market)
            if self.train(market):
                after = "<nouveau_lot>" if self.args.dry_run else self.trained_stamp(market)
                if after and after != before:
                    stamp = after
                else:
                    self._fail(market, "entraînement sans lot nommé, le lot en place reste chargé", halt=False)
            else:
                self._fail(market, "réentraînement échoué, le lot en place reste chargé", halt=False)
        elif not decision.guard.get("allowed"):
            # Séance manquante ou incomplète : le lot reste, le journal tourne sur les données complètes.
            held = " — réentraînement reporté" if decision.wanted else ""
            self.outcome.stale.append(f"{market} : {'; '.join(decision.guard.get('reasons') or [])}{held}")
            print(f"[AVERTISSEMENT] {label} : {decision.summary()}")
        if not stamp:
            self._fail(market, "aucun lot en place ni nouveau lot")
            return
        new_lot = bool(stamp != in_place)
        if self.run(f"{label} : prévisions journalisées ({stamp})", self.journal_cmd(market, stamp)) != 0:
            self._fail(market, "journal non écrit ; relancer reprend le même identifiant")
            return
        manifest = [self.py, "scripts/write_experimental_release.py", "--market", market, "--stamp", stamp]
        if self.run(f"{label} : manifeste expérimental", manifest) != 0:
            self._fail(market, "manifeste expérimental non écrit")
            return
        if new_lot:
            self.outcome.new_lots[market] = stamp
        self.serve(market, recreate=new_lot)

    def collect(self, market: str) -> bool:
        if market == "cocoa":
            code = self.run("Cacao : ICE London (Databento puis Playwright)", [self.py, "collect_ice_london_cocoa.py"])
            if code == 1:
                self._fail(market, "collecte ICE London échouée")
                return False
            if code >= 2:
                self.outcome.stale.append("cocoa : source indisponible ou base en retard")
            self._optional("Cacao : courbe Londres mois nommés", [self.py, "collect_london_named_futures.py"])
            if self.run("Cacao : courbe NY (Investing)", [self.py, "collect_cocoa_futures_investing.py"]) != 0:
                self._optional("Cacao : courbe NY (repli Yahoo)", [self.py, "collect_futures.py"])
            self._optional("Cacao : météo des régions", [self.py, "collect_cocoa_weather.py"])
            self._optional("News et sentiment", [self.py, "collect_news.py"])
            self._optional("Événements NLP", [self.py, "scripts/record_news_events.py"])
            if self.run("Cacao : série sans saut (Databento)", [self.py, "scripts/build_cocoa_roll_series.py"]) != 0:
                self._optional("Cacao : série sans saut depuis le brut local", [self.py, "scripts/build_cocoa_roll_series.py", "--offline"])
            return True
        for attempt in range(1, ROBUSTA_COLLECT_TRIES + 1):
            code = self.run("Robusta : Databento RC.c.0..2 et série sans saut", [self.py, "collect_coffee_robusta_price.py"])
            if code == 0:
                return True
            if code >= 4:
                self.outcome.quarantine.append("coffee_robusta : prix non daté ou jour fermé, rien d'écrit")
                return False
            if code == 1:
                self.outcome.partial.append("coffee_robusta : collecte échouée, le cacao reste consultable")
                return False
            if attempt < ROBUSTA_COLLECT_TRIES:
                print(f"[AVERTISSEMENT] Source robusta indisponible ({attempt}/{ROBUSTA_COLLECT_TRIES}), nouvel essai dans {ROBUSTA_RETRY_SECONDS} s")
                self.sleep(ROBUSTA_RETRY_SECONDS)
        self.outcome.stale.append("coffee_robusta : source indisponible après 3 essais")
        return False

    def decide(self, market: str, until: str):
        from src.models import retrain_policy as policy

        snapshot = self.root / SNAPSHOT[market]
        stamp = _read_json(snapshot.with_suffix(".json")) or {}
        freshness = _read_json(self.root / "data" / "status" / f"freshness_{market}.json")
        guard = policy.training_guard(freshness, stamp.get("raw_max_date") or stamp.get("max_date"), until)
        version = self.lot_in_place(market)
        trained = policy.lot_trained_on(version)
        shock = policy.market_shock(self._adjusted_closes(market, until), trained)
        degraded = policy.degradation(*self._matured_inputs(market), excluded=policy.contaminated_versions())
        decision = policy.decide(
            market,
            today=self.now.date(),
            lot_version=version,
            guard=guard,
            force=bool(self.args.retrain),
            weekly=self.args.mode == "weekly",
            shock=shock,
            degraded=degraded,
        )
        self.decisions[market] = decision
        print(f"[DÉCISION] {LABEL[market]} : {decision.summary()}")
        if not self.args.dry_run:
            policy.write_decision(decision, self.root / "data" / "status")
        return decision

    def _adjusted_closes(self, market: str, until: str):
        """Clôtures de la série sans saut : un roulement n'y crée pas de rendement."""
        import pandas as pd

        path = self.root / ROLL_SERIES[market]
        if not path.exists():
            return None
        frame = pd.read_csv(path, usecols=["date", "price"])
        frame = frame[frame["date"].astype(str).str[:10] <= until]
        return pd.Series(frame["price"].to_numpy(), index=pd.to_datetime(frame["date"]))

    def _matured_inputs(self, market: str):
        import pandas as pd

        from src.models.market_registry import get_market_config
        from src.models.prediction_journal import load_forecasts

        names = set(get_market_config(market).api_markets)
        rows = load_forecasts(self.root / "data" / "prediction_journal.jsonl")
        actuals: dict[str, float] = {}
        snapshot = self.root / SNAPSHOT[market]
        if snapshot.exists():
            frame = pd.read_csv(snapshot, usecols=["date", "price"])
            actuals = {str(d)[:10]: float(p) for d, p in zip(frame["date"], frame["price"])}
        return rows, actuals, names

    def train(self, market: str) -> bool:
        label = LABEL[market]
        snapshot = SNAPSHOT[market]
        cmd = [
            self.py, "train_hybrid_improved.py", "--market", market,
            "--snapshot", snapshot, "--calendar", MARKET_CALENDAR[market],
        ]
        if self.run(f"{label} : entraînement du lot nommé", cmd) != 0:
            return False
        if market != "coffee_robusta":
            return True
        nhits = [
            "python", "-u", "train_nhits.py", "--market", market,
            "--snapshot", snapshot, "--calendar", MARKET_CALENDAR[market],
        ]
        if self.run("Robusta : N-HiTS (Docker)", ["docker", "compose", "exec", "-T", "api", *nhits], env=DOCKER_ENV) == 0:
            return True
        retry = ["docker", "compose", "run", "--rm", "--no-deps", "api", *nhits]
        return self.run("Robusta : N-HiTS (conteneur ponctuel)", retry, env=DOCKER_ENV) == 0

    def journal_cmd(self, market: str, stamp: str, preview: bool = False) -> list[str]:
        tail = ["--market", market, "--snapshot", SNAPSHOT[market], "--stamp", stamp]
        if preview:
            tail += ["--preview", "--preview-output", f"{PREVIEW_DIR}/preview_journal_{market}.json"]
        if market == "coffee_robusta":
            mount = f"{self.root / 'scripts'}:/app/scripts:ro"
            return [
                "docker", "compose", "run", "--rm", "--no-deps", "-v", mount, "api",
                "python", "-u", "scripts/journal_forecast_revisions.py", *tail,
            ]
        return [self.py, "scripts/journal_forecast_revisions.py", *tail]

    def serve(self, market: str, *, recreate: bool) -> None:
        label = LABEL[market]
        up = ["docker", "compose", "up", "-d"] + (["--force-recreate"] if recreate else []) + ["api"]
        what = "recréée (nouveau lot)" if recreate else "en place (même lot, données relues)"
        if self.run(f"{label} : API locale {what}", up, env=DOCKER_ENV) != 0:
            self.outcome.partial.append(f"{market} : API non démarrée")
            return
        if not self.wait_for_api(recreate):
            self.outcome.partial.append(f"{market} : API non prête, lot non confirmé")
            return
        verify = [self.py, "scripts/verify_served_forecast.py", "--market", market]
        code = self.run(f"{label} : contrôle du lot servi, J+1 J+7 J+14 J+30", verify, env=self._token_env())
        if code != 0:
            self.outcome.partial.append(f"{market} : lot servi non confirmé (code {code})")

    def wait_for_api(self, recreated: bool) -> bool:
        """Seul un 200 compte : un 503 signale un service dégradé."""
        if self.args.dry_run:
            print(f"[ÉTAPE] Attente de {HEALTH_URL} (200 seulement)")
            return True
        if recreated:
            print("[INFO] Premier démarrage après recréation : chargement des modèles (jusqu'à 10 minutes)")
            self.sleep(20)
        for attempt in range(1, 61):
            code = self.probe()
            if code == 200:
                print("[OK] API démarrée, /health répond 200")
                return True
            print(f"[INFO] API pas encore prête ({code or 'injoignable'}) {attempt}/60")
            self.sleep(10)
        print("[AVERTISSEMENT] API non prête après 10 minutes : docker logs cocoa-api --tail 80")
        return False

    def after_markets(self) -> None:
        trained = set(self.outcome.new_lots) | ({*self.args.markets} if self.args.mode == "weekly" else set())
        for market in [m for m in self.args.markets if m in trained]:
            wf = [
                self.py, "scripts/run_walk_forward_validation.py", "--market", market,
                "--max-origins", "24", "--skip-nhits", "--horizons", "1", "7", "14", "30",
                "--origin-window", "recent", "--direct-hstep", "--skip-calibration", "--measure-only",
                "--output-dir", WALK_FORWARD_DIR[market],
            ]
            self._optional(f"{LABEL[market]} : walk-forward (mesure seulement)", wf)
        if trained:
            if self.run("Cacao : courbe à terme Londres", [self.py, "train_futures_curve.py", "--source", "london"]) != 0:
                self._optional("Cacao : courbe à terme (repli Investing)", [self.py, "train_futures_curve.py", "--source", "investing"])
            self._optional("Cacao : courbe Londres mois nommés", [self.py, "train_futures_curve.py", "--source", "london_named"])
            self._optional("Nettoyage des anciens modèles (lots des manifestes protégés)", [self.py, "scripts/cleanup_old_models.py", "--keep", "5"])
        self._optional("Évaluation des prévisions échues", [self.py, "evaluate_predictions.py"])
        self.deploy()
        code = self.run("Fraîcheur des prix", [self.py, "-m", "src.monitoring.freshness", *self.args.markets])
        if code == 2:
            self.outcome.stale.append("fraîcheur : séance attendue manquante")

    def deploy(self) -> None:
        if os.name != "nt" and not self.args.dry_run:
            print("[INFO] Déploiement VPS : hors Windows, prévu par git au lot C")
            return
        base = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.root / "deploy_models.ps1")]
        if self.outcome.new_lots:
            label, cmd = "VPS : déploiement complet (manifeste actif, code, données)", base
        else:
            label, cmd = "VPS : données seules (séries sans saut, sans redémarrage)", base + ["-DataOnly"]
        if self.run(label, cmd) != 0:
            self.outcome.partial.append("VPS non mis à jour ; l'API locale est à jour")

    def preview(self) -> int:
        """Prévisions du lot en place, affichées sans rien écrire dans le journal."""
        code = EXIT_OK
        for market in self.args.markets:
            stamp = stamp_of(self.lot_in_place(market))
            if not stamp:
                print(f"[ERREUR] {LABEL[market]} : aucun lot en place")
                code = EXIT_FAILED
                continue
            env = DOCKER_ENV if market == "coffee_robusta" else None
            label = f"{LABEL[market]} : aperçu du journal ({stamp}), rien n'est écrit"
            if self.run(label, self.journal_cmd(market, stamp, preview=True), env=env) != 0:
                code = EXIT_FAILED
        return code

    # ------------------------------------------------------------------ outils
    def _optional(self, label: str, cmd: list[str]) -> int:
        code = self.run(label, cmd)
        if code != 0:
            print(f"[AVERTISSEMENT] {label} : non bloquant (code {code})")
        return code

    def _fail(self, market: str, reason: str, *, halt: Optional[bool] = None) -> None:
        """Le cacao arrête le traitement ; le robusta ne retire jamais le cacao déjà servi."""
        print(f"[ERREUR] {LABEL[market]} : {reason}" if market == "cocoa" else f"[AVERTISSEMENT] {LABEL[market]} : {reason}")
        if market == "cocoa":
            self.outcome.failed.append(f"cocoa : {reason}")
            if halt is None or halt:
                self.outcome.halted = True
        else:
            self.outcome.partial.append(f"coffee_robusta : {reason}")

    def _api_token(self) -> str:
        token = os.environ.get("API_TOKEN", "")
        if token:
            return token
        try:
            from dotenv import dotenv_values

            return str(dotenv_values(self.root / ".env").get("API_TOKEN") or "")
        except Exception:
            return ""

    def _token_env(self) -> dict:
        token = self._api_token()
        return {"API_TOKEN": token} if token else {}

    def finish(self) -> int:
        code = self.outcome.code()
        messages = {
            EXIT_OK: "MISE À JOUR TERMINÉE",
            EXIT_FAILED: "ÉCHEC — voir les [ERREUR] ci-dessus",
            EXIT_STALE: "MISE À JOUR AVEC DONNÉES EN RETARD OU SÉANCE INCOMPLÈTE",
            EXIT_QUARANTINE: "MISE À JOUR AVEC QUARANTAINE — un prix non daté n'a pas été écrit",
            EXIT_PARTIAL: "MISE À JOUR PARTIELLE — voir les [AVERTISSEMENT] ci-dessus",
        }
        print(f"\n{'=' * 80}")
        for kind in ("failed", "stale", "quarantine", "partial"):
            for item in getattr(self.outcome, kind):
                print(f"  [{kind}] {item}")
        for market, stamp in self.outcome.new_lots.items():
            print(f"  [nouveau lot expérimental] {market} {stamp} — non promu, validation manuelle")
        prefix = "[DRY-RUN] " if self.args.dry_run else ""
        print(f"  {prefix}{messages[code]}\n  Code de sortie : {code}\n{'=' * 80}")
        return code


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("daily", "weekly"), default="daily")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner même sans déclencheur")
    parser.add_argument("--dry-run", action="store_true", help="Lister les étapes sans rien lancer ni écrire")
    parser.add_argument("--preview-journal", action="store_true", help="Prévisions du lot en place, sans écriture")
    parser.add_argument("--unattended", action="store_true", help="Aucune pause (lu par update_system.bat)")
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 80)
    print(f"  TRAITEMENT {args.mode.upper()} — {', '.join(LABEL[m] for m in args.markets)}")
    print("=" * 80)
    return Pipeline(args, Runner(dry_run=args.dry_run)).execute()


if __name__ == "__main__":
    raise SystemExit(main())
