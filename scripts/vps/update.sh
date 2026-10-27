#!/usr/bin/env bash
# Met /opt/prediction sur un commit de origin/main par git, au lieu des copies de deploy_models.ps1.
#
# Usage, depuis /opt/prediction, après « git fetch origin » :
#   scripts/vps/update.sh --dry-run <commit>   inventaire seul, rien n'est écrit
#   scripts/vps/update.sh <commit>             sauvegarde, puis git checkout du commit
# Premier passage, quand ce script n'est pas encore sur le disque :
#   git show <commit>:scripts/vps/update.sh | bash -s -- --dry-run <commit>
#
# Chaque fichier que git remplacerait est classé :
#   garde    manifeste expérimental écrit par le traitement du VPS : remis tel quel après ;
#   crlf     même contenu que le commit, aux fins de ligne Windows près (copie de deploy_models.ps1) ;
#   ancien   version présente dans l'historique git : rien d'unique n'est perdu ;
#   UNIQUE   contenu absent de git : refus (code 3), à examiner avant toute mise à jour.
# Fichiers suivis par HEAD que la cible ne suit plus (88 modèles de mai à juillet pour bcab53e) :
# git les effacerait. Ils sont mis de côté puis remis sur le disque quand la cible les ignore
# (models/, data/...) ; les autres restent retirés, dans la sauvegarde.
# Tout ce qui est remplacé ou retiré part d'abord dans ../prediction-backups/. .env et les
# fichiers ignorés ne sont pas touchés. Ni build ni redémarrage : les commandes à lancer
# ensuite sont affichées, à lancer après accord.

set -euo pipefail
export LC_ALL=C

src="${BASH_SOURCE[0]:-}"
if [[ -n "$src" && -f "$src" ]]; then
  ROOT="$(cd "$(dirname "$src")/../.." && pwd)"
else
  ROOT="$(pwd)"
fi
cd "$ROOT"
[[ -d .git ]] || { echo "[ERREUR] $ROOT n'est pas un dépôt git" >&2; exit 1; }
BACKUP_DIR="${BACKUP_DIR:-$(dirname "$ROOT")/prediction-backups}"
KEEP_RE='^config/(previous_)?experimental_release_[a-z_]+\.json$'

dry_run=0
target=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -*) echo "option inconnue : $arg" >&2; exit 1 ;;
    *) target="$arg" ;;
  esac
done
[[ -n "$target" ]] || { echo "usage : $0 [--dry-run] <commit>" >&2; exit 1; }
if ! git cat-file -e "$target^{commit}" 2>/dev/null; then
  echo "[ERREUR] commit $target absent : lancer « git fetch origin » d'abord" >&2
  exit 1
fi
target="$(git rev-parse "$target^{commit}")"
if ! git merge-base --is-ancestor "$target" origin/main; then
  echo "[ERREUR] $target n'est pas sur origin/main" >&2
  exit 1
fi

mkdir -p logs/pipeline
exec 9>logs/pipeline/.lock
if ! flock -n 9; then
  echo "[ERREUR] un traitement tourne : mise à jour refusée" >&2
  exit 1
fi

in_history() {  # le blob $2 du fichier $1 existe-t-il dans un commit ?
  local commit
  for commit in $(git log --all --format=%H -- "$1"); do
    [[ "$(git rev-parse -q --verify "$commit:$1" 2>/dev/null || true)" == "$2" ]] && return 0
  done
  return 1
}

head="$(git rev-parse HEAD)"
echo "HEAD  $(git log -1 --format='%h %cs %s' HEAD)"
echo "cible $(git log -1 --format='%h %cs %s' "$target")"

# Fichiers suivis modifiés, et fichiers non suivis que le commit cible apporterait.
list="$(mktemp)"
gone="$(mktemp)"
trap 'rm -f "$list" "$gone"' EXIT
{
  git diff --name-only HEAD
  comm -12 <(git ls-tree -r --name-only "$target" | sort) <(git ls-files --others | sort)
} | sort -u >"$list"
# Suivis par HEAD, plus par la cible, présents sur le disque : git les effacerait.
git diff --name-only --diff-filter=D HEAD "$target" | while IFS= read -r file; do
  [[ -e "$file" ]] && printf '%s\n' "$file"
done >"$gone"

keep=()
unique=()
declare -A count=([garde]=0 [crlf]=0 [ancien]=0 [UNIQUE]=0 [absent]=0)
while IFS= read -r file; do
  [[ -n "$file" ]] || continue
  if [[ ! -e "$file" ]]; then
    kind="absent"
  elif [[ "$file" =~ $KEEP_RE ]]; then
    kind="garde"
    keep+=("$file")
  else
    blob="$(tr -d '\r' <"$file" | git hash-object --stdin)"
    if [[ "$blob" == "$(git rev-parse -q --verify "$target:$file" 2>/dev/null || true)" ]]; then
      kind="crlf"
    elif in_history "$file" "$blob" || in_history "$file" "$(git hash-object "$file")"; then
      kind="ancien"
    else
      kind="UNIQUE"
      unique+=("$file")
    fi
  fi
  count[$kind]=$(( ${count[$kind]} + 1 ))
  echo "  $kind $file"
done <"$list"
echo "bilan : garde=${count[garde]} crlf=${count[crlf]} ancien=${count[ancien]} UNIQUE=${count[UNIQUE]} absent=${count[absent]}"
echo "plus suivis par la cible : $(wc -l <"$gone") fichier(s), remis sur le disque si la cible les ignore :"
cut -d/ -f1 "$gone" | sort | uniq -c | sed 's/^/  /'
echo "image et compose (HEAD -> cible) :"
git diff --stat "$head" "$target" -- requirements.txt Dockerfile docker-compose.yml .dockerignore | sed 's/^/  /'

if (( ${#unique[@]} )); then
  echo "[REFUS] contenu absent de git : ${unique[*]}"
  echo "        à examiner (et à commiter s'il doit rester) avant la mise à jour."
  exit 3
fi
if (( dry_run )); then
  echo "[DRY-RUN] rien n'est écrit."
  exit 0
fi

stamp="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
archive="$BACKUP_DIR/update_${stamp}_${head:0:7}.tgz"
cat "$list" "$gone" | tar czf "$archive" --ignore-failed-read -T -
echo "[OK] sauvegarde $archive"
saved="$BACKUP_DIR/keep_${stamp}"
while IFS= read -r file; do
  mkdir -p "$saved/$(dirname "$file")"
  cp -p "$file" "$saved/$file"
done < <(printf '%s\n' "${keep[@]}" | sed '/^$/d'; cat "$gone")

git checkout -q -f -B main "$target"
for file in "${keep[@]}"; do
  cp -p "$saved/$file" "$file"
done
kept=0
dropped=0
while IFS= read -r file; do
  if git check-ignore -q -- "$file"; then
    mkdir -p "$(dirname "$file")"
    cp -p "$saved/$file" "$file"
    kept=$((kept + 1))
  else
    echo "  retiré (dans la sauvegarde) : $file"
    dropped=$((dropped + 1))
  fi
done <"$gone"
echo "[OK] $(git log -1 --format='%h %s')"
echo "[OK] plus suivis : $kept remis sur le disque (ignorés), $dropped retirés"
git status --short --untracked-files=no | sed 's/^/  reste modifié : /'

echo "Ensuite, après accord :"
echo "  - requirements.txt, Dockerfile ou .dockerignore changés : docker compose build api"
echo "  - code, config, compose ou image changés : (unset PREDICTION_RELEASE_MODE; docker compose up -d --no-deps --force-recreate api)"
echo "    (src/ et config/ sont montés : sans --force-recreate, compose répond « Running » et l'API garde l'ancien code)"
echo "    puis curl -s localhost:8000/health | jq '{release_mode, serving}' : mode active, lot actif"
