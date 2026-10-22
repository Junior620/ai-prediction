#!/usr/bin/env bash
# Traitement du soir sur le VPS : scripts/run_pipeline.py --host vps dans un conteneur
# ponctuel de l'image de l'API (docker compose run --rm --no-deps api).
#
# Usage : scripts/vps/pipeline.sh [--catch-up] [arguments de run_pipeline.py]
#   --catch-up  relance le cacao seulement si le dernier passage a fini en code 2 à cause
#               du cacao (séance pas encore publiée par Databento) ; sinon ne fait rien.
#   --dry-run   (passé à run_pipeline.py) liste les étapes ; le dernier code n'est pas noté.
#
# Jamais de « docker compose up », « restart » ni « build » : l'API publique garde son
# conteneur, son mode active et son lot actif. Le mode vient de .env, jamais du shell.
# Codes : ceux de run_pipeline.py (0 succès, 1 échec, 2 retard ou séance incomplète,
# 3 quarantaine, 4 partiel). Conteneur impossible à lancer : 1, noté dans syslog.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="$ROOT/logs/pipeline"
STATUS="$ROOT/data/status/pipeline_last_run.json"
IMAGE="$(basename "$ROOT")-api"
mkdir -p "$LOG_DIR"

catch_up=0
if [[ "${1:-}" == "--catch-up" ]]; then
  catch_up=1
  shift
fi
dry_run=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && dry_run=1
done

if (( catch_up )); then
  # jq lit le dernier passage écrit par run_pipeline.py ; rien à rattraper sinon.
  if ! jq -e '.code == 2 and ([.stale[] | select(startswith("cocoa") or startswith("API publique"))] | length > 0)' \
      "$STATUS" >/dev/null 2>&1; then
    exit 0
  fi
  set -- --markets cocoa "$@"
fi

LOG="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"
alert() {
  echo "[ERREUR] $1" >>"$LOG"
  logger -t prediction-pipeline -p user.err "$1 (voir $LOG)"
}

exec 9>"$LOG_DIR/.lock"
if ! flock -n 9; then
  alert "un traitement ou une mise à jour tourne déjà, passage ignoré"
  exit 1
fi
if ! grep -qx 'PREDICTION_RELEASE_MODE=active' .env; then
  alert ".env ne contient pas la ligne PREDICTION_RELEASE_MODE=active, rien n'est lancé"
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  # Une image absente serait construite en silence par compose : refusé, la construction se décide.
  alert "image $IMAGE absente, rien n'est lancé (reconstruction après accord, docs/VPS_PIPELINE.md)"
  exit 1
fi
unset PREDICTION_RELEASE_MODE

echo "[INFO] $(date -u +%FT%TZ) début, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'), arguments : $*" >>"$LOG"
# PREDICTION_JOURNAL vide : le traitement écrit toujours data/prediction_journal.jsonl, même si
# .env envoie l'API publique vers son propre journal (prediction_journal_public.jsonl).
docker compose run --rm --no-deps -T \
  -v "$ROOT:/app" -e PIPELINE_HOST=vps -e PREDICTION_JOURNAL= \
  api python -u scripts/run_pipeline.py --host vps --unattended --alert "$@" >>"$LOG" 2>&1
code=$?
echo "[INFO] $(date -u +%FT%TZ) fin, code $code" >>"$LOG"

case "$code" in
  0) ;;
  1|2|3|4) logger -t prediction-pipeline -p user.warning "code $code (voir $LOG)" ;;
  *) alert "conteneur du traitement en échec (code $code)"; code=1 ;;
esac
(( dry_run )) || echo "$code" >"$LOG_DIR/last_exit_code"
find "$LOG_DIR" -name '*.log' -mtime +90 -delete 2>/dev/null
exit "$code"
