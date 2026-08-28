#!/usr/bin/env bash
# Installe Playwright + crons collecte prix sur le VPS Contabo.
# Usage: bash scripts/setup_vps_price_cron.sh

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/prediction}"
PYTHON="${APP_DIR}/venv_py311/bin/python"
PIP="${APP_DIR}/venv_py311/bin/pip"

echo "==> Installation dependances Playwright dans ${APP_DIR}"
cd "${APP_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERREUR: venv introuvable (${PYTHON})"
  exit 1
fi

"${PIP}" install -q 'playwright>=1.41.0'
"${PYTHON}" -m playwright install chromium
# Deps systeme (Ubuntu/Debian) — peut necessiter sudo
if command -v sudo >/dev/null 2>&1; then
  sudo "${PYTHON}" -m playwright install-deps chromium || true
else
  "${PYTHON}" -m playwright install-deps chromium || true
fi

CRON_FILE="/tmp/prediction_price_cron.txt"
cat > "${CRON_FILE}" <<EOF
# Collecte prix cacao ICE London — lun-ven 17:35 UTC (apres cloture ICE London)
35 17 * * 1-5 cd ${APP_DIR} && ${PYTHON} collect_ice_london_cocoa.py >> ${APP_DIR}/logs/ice_london_cron.log 2>&1
EOF

echo "==> Entrees cron proposees:"
cat "${CRON_FILE}"
echo ""
echo "Pour installer: crontab -l 2>/dev/null | cat - ${CRON_FILE} | crontab -"
echo "Ou copier manuellement les lignes ci-dessus dans crontab -e"
