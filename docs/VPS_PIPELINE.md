# Traitement du soir sur le VPS (lot C)

Le VPS (`root@169.58.99.28`, `/opt/prediction`) collecte, entraîne et journalise seul avec
`scripts/run_pipeline.py --host vps`. Il le fait dans un conteneur ponctuel de l'image de l'API.
L'API publique reste en mode `active` : jamais de `docker compose up`, `restart` ni `build` par le
traitement, et jamais d'écriture de `config/active_release_*.json`.

Rien de ce document n'est encore appliqué sur le VPS. Chaque étape de la bascule (plus bas)
attend l'accord du propriétaire.

## État du VPS au 20/10/2026 (lecture seule)

| Point | Constat | Conséquence |
|---|---|---|
| Dépôt | git sur `bcab53e` (26/08), ancêtre de `main`, 26 commits de retard. 25 fichiers suivis modifiés, 232 non suivis (179 modèles, 32 rapports, 15 `src/`, 5 `config/`, 1 `.env.bak-20261020`). | Mise à jour par git possible en avance rapide. |
| Fichiers modifiés | 77 fichiers hors modèles comparés à `main`, retours chariot retirés. 72 sont identiques à `main` : ce sont des copies CRLF faites par `deploy_models.ps1`. Les 5 autres sont des versions plus anciennes présentes dans git : `config.yaml` (10/09), `horizon_schedule.py` (13/10), `hybrid_features.py`, `prediction_journal.py` et `revision_command.py` (20/10). | Aucun contenu propre au VPS : un `git checkout` ne perd rien. `update.sh` sauvegarde quand même. |
| Modèles suivis | 88 fichiers de `models/` (lots de mai à juillet) sont suivis par `bcab53e`. `main` ne les suit plus. | Un `git checkout` simple les effacerait. `update.sh` les met de côté, puis les remet sur le disque, où ils deviennent ignorés. |
| `.env.bak-20261020` | Non suivi. | Ignoré par `.env*` du `.gitignore` de `main`. |
| Conteneur `cocoa-api` | Créé le 01/09 avec un ancien compose. Il monte `config`, `logs`, `models`, `reports`, `src` et `.env` (lecture seule). Il ne monte ni `data/`, ni `scripts/`, ni `train_nhits.py`. Aucune variable `PREDICTION_RELEASE_MODE` : le mode `active` vient de la valeur par défaut. | La série envoyée par `-DataOnly` n'atteint pas l'API (`/health` : `roll_series_end` nul). Le journal servi par l'API vit dans le conteneur (`/app/data/prediction_journal.jsonl`, 343 Ko) et disparaît à la recréation s'il n'est pas copié avant. |
| Image `prediction-api:latest` | Construite le 30/07, contient `/app/.env`. Vérifié le 20/10 : `databento`, `feedparser`, `langdetect`, `deep-translator` et `playwright` sont absents, et `neuralforecast` ne s'importe pas. | La collecte Databento échouerait : reconstruction nécessaire avant le premier vrai passage, après accord. |
| Python hôte | 3.12, sans pandas, sans venv. | Tout tourne dans l'image de l'API. |
| Outils | Docker 29.6, Compose 5.3, `flock`, `jq` et `logger` présents. 6 CPU, 11 Gio de RAM, 173 Go libres. | |
| Heure | Fuseau Europe/Berlin. Le cron d'Ubuntu ignore `CRON_TZ`. | Horaires du cron en heure de Berlin (voir plus bas). Le `CRON_TZ=Africa/Douala` du cron `/opt/samen` est aussi ignoré : il tourne à 08:00 Berlin. Ce cron n'est pas touché. |
| Données | `data/` ne contient que `cocoa_roll_series.csv`. Ni journal, ni instantanés, ni état, ni météo, ni événements, ni calendrier Databento, ni `models/candidates/`. | Amorçage unique depuis le PC avant que le VPS écrive (étape 4). |
| GitHub | `git ls-remote origin` fonctionne sans identifiants et voit `main` = `5fde694`. | `git fetch` possible depuis le VPS. |

## Exécution sur le VPS

```
scripts/vps/pipeline.sh
  └─ docker compose run --rm --no-deps -T -v /opt/prediction:/app api \
       python -u scripts/run_pipeline.py --host vps --unattended --alert
```

- **Conteneur ponctuel.** Il utilise l'image de l'API, ses variables (`.env`, jamais affiché) et son réseau (`redis`, `api`). Le dépôt entier est monté sur `/app`. Chaque étape tourne avec le Python du conteneur, sans aucune commande docker.
- **Différences avec le PC.**
  - Pas de `deploy_models.ps1` : le VPS est la cible.
  - Le N-HiTS et le journal du robusta tournent directement dans le conteneur.
  - Pas d'`API_TOKEN` : pas d'appel à `/predict`.
- **API publique.** Après l'évaluation, le cache Redis est vidé (même effet que `-DataOnly`), puis `/health` est contrôlé :

  | Constat sur `/health` | Code |
  |---|---|
  | `release_mode` différent de `active` | 1 |
  | Lot servi différent du manifeste actif, pour un marché | 1 |
  | Dernière séance du cacao antérieure au jour figé | 2 |
  | Série sans saut antérieure au jour figé (`./data` non monté) | 4 |
  | API injoignable après 6 essais d'une minute | 4 |

- **Gardes du wrapper.** Il refuse de lancer si :
  - un autre traitement tourne (`flock`) ;
  - `.env` n'a pas la ligne `PREDICTION_RELEASE_MODE=active` ;
  - l'image est absente (compose la construirait en silence).

  Il retire `PREDICTION_RELEASE_MODE` du shell et force `PREDICTION_JOURNAL` à vide : le traitement écrit toujours `data/prediction_journal.jsonl`.
- **Un seul hôte écrit.** `config/config.yaml`, `pipeline.journal_writer` (`pc` par défaut, `vps` depuis le 27/10/2026). L'autre hôte refuse le traitement réel (code 1). `--dry-run` et `--preview-journal` restent permis partout.

## Codes de sortie et alertes

| Code | Sens | Alerte |
|---|---|---|
| 0 | Succès | Aucune |
| 1 | Échec réel : collecte cacao, instantané, série sans saut, entraînement, journal, manifeste. Ou API publique hors mode `active`, mauvais lot servi, hôte non autorisé, verrou, `.env`, image ou conteneur. | CRITICAL |
| 2 | Source en retard ou séance incomplète | ERROR |
| 3 | Quarantaine : prix non daté ou jour fermé, rien d'écrit | ERROR |
| 4 | Succès partiel | ERROR |

Où arrivent les alertes :

- **Webhook ou SMTP**, par `src/monitoring/alert_system.py`. Il faut `ALERT_WEBHOOK_URL`, ou `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` et `ALERT_SMTP_HOST` dans le `.env` du VPS. Un webhook Discord (`https://discord.com/api/webhooks/…`) reçoit son format natif ; les autres reçoivent le format Slack.
- **Webhook aussi pour `pipeline.sh`** : verrou pris, `.env` sans mode `active`, image absente ou conteneur en échec. Ces cas ne passent pas par Python ; le script poste lui-même sur `ALERT_WEBHOOK_URL` (pas de SMTP).
- **Syslog** : `journalctl -t prediction-pipeline`, toujours.
- **Test** : `scripts/vps/pipeline.sh --test-alert` envoie un message par chacun des deux chemins et ne lance rien d'autre.

Où lire le résultat :

- `data/status/pipeline_last_run.json` : hôte, code, retards et nouveaux lots ;
- `logs/pipeline/<date UTC>.log` : un fichier par passage, gardé 90 jours.

Depuis le lot B, le robusta vient de Databento. La collecte ne demande que la dernière semaine
(0,004 $) et a besoin du brut versionné `data/databento_robusta_calendar_raw.csv` (historique
téléchargé une fois, 1,226 $). Sur l'hôte qui écrit le journal, ce fichier doit être amorcé
depuis le PC, comme le reste de `data/`. Sans lui, le robusta finit en code 2
(« historique absent ») et envoie une alerte. D'ici là, on peut ajouter `--markets cocoa` à
la ligne du cron. C'est au propriétaire de décider.

## Cron (installé le 27/10/2026, déplacé au matin le 28/10/2026)

`scripts/vps/prediction-pipeline.cron`, installé sous un nom sans point. Le cron ignore les noms
avec un point dans `/etc/cron.d`.

```bash
install -o root -g root -m 644 scripts/vps/prediction-pipeline.cron /etc/cron.d/prediction-pipeline
```

| Ligne (heure de Berlin) | UTC été | UTC hiver | Séance |
|---|---|---|---|
| `30 6 * * 2-6` | 04:30 | 05:30 | La veille, du lundi au vendredi. Le samedi matin est un samedi UTC : la politique entraîne le lot hebdomadaire. |
| `30 9 * * 2-6 … --catch-up` | 07:30 | 08:30 | Ne relance que le marché (cacao ou robusta) dont la séance manquait encore au passage du matin (code 2). |

Pourquoi le matin : Databento publie la séance de Londres dans la nuit, pas à la clôture. Le
27/10/2026 à 23:30 UTC, la séance du jour n'était pas encore publiée (fin de plage au 27/10
00:00). Et la borne « jamais au-delà d'aujourd'hui (UTC) » écarte de toute façon la séance du
jour même. Le premier cron, à 00:30 Berlin, ne prenait donc jamais la séance qui venait de fermer :
le site affichait « retard d'une séance » toute la journée. Un test vérifie l'horaire en été et en
hiver. Il reste une fenêtre courte, de minuit (Londres) au passage du matin, où la séance de la
veille est attendue mais pas encore traitée.

## Qui écrit `data/prediction_journal.jsonl` : décision à prendre

| | A. Le VPS écrit (recommandé) | B. Le PC reste l'écrivain |
|---|---|---|
| Traitement | Le VPS tourne chaque matin, sans le PC allumé. | Le PC continue comme aujourd'hui. |
| Journal et lots | Écrits au même endroit que l'API publique. | Écrits sur le PC. |
| Côté PC | Lecteur seul : `update_system.bat` refuse le traitement réel. Aperçu et dry-run restent permis. | Inchangé. |
| Comment l'autre reçoit | Le PC rapatrie par `scp` (VPS vers PC, jamais l'inverse) : `data/prediction_journal.jsonl`, `data/status/`, `data/freeze/`, `config/experimental_release_*.json`, `reports/walk_forward/*/daily/`. Script à écrire après la décision. | Le VPS ne reçoit que `deploy_models.ps1`. |
| Lot C | Utile. | Ne sert qu'en secours. Ne pas installer le cron : il refuserait chaque nuit avec une alerte. |

**Journal de l'API publique.** L'API journalise aussi ce qu'elle sert : lignes `release_mode = active`,
lues par `/tracking`. Pour ne pas les mêler au journal expérimental du traitement, que lit la
mesure de dégradation, le `.env` du VPS doit envoyer l'API vers son propre fichier :
`PREDICTION_JOURNAL=/app/data/prediction_journal_public.jsonl`. Ce fichier est amorcé avec le
journal qui vit aujourd'hui dans le conteneur (étape 0). Le wrapper force la valeur à vide pour
le traitement.

## Bascule, étape par étape (chacune après accord)

### 0. Garder le journal public avant toute recréation

```bash
docker cp cocoa-api:/app/data/prediction_journal.jsonl /opt/prediction/data/prediction_journal_public.jsonl
```

Puis le propriétaire ajoute `PREDICTION_JOURNAL=/app/data/prediction_journal_public.jsonl` au
`.env` du VPS.

### 1. Code par git

`<sha>` est le commit de `main` après fusion de cette PR.

```bash
git fetch origin
git show <sha>:scripts/vps/update.sh | bash -s -- --dry-run <sha>
git show <sha>:scripts/vps/update.sh | bash -s -- <sha>
```

Avant tout changement, les fichiers remplacés sont archivés dans `/opt/prediction-backups/`.

Répétition faite le 20/10/2026 sur une copie de `/opt/prediction` dans `/tmp` (sans `.env`, `data/` ni `logs/`), vers `5fde694` :

- bilan : 72 copies CRLF, 5 versions plus anciennes, 0 contenu propre au VPS ;
- 88 modèles plus suivis, tous remis sur le disque ;
- ensuite : aucun fichier suivi modifié, aucun fichier non suivi visible.

La copie a ensuite été supprimée. `/opt/prediction` n'a pas été modifié.

### 2. Image et conteneur

```bash
docker compose build api
(unset PREDICTION_RELEASE_MODE; docker compose up -d --no-deps api)
```

Ensuite, vérifier :

- `/health` : `release_mode` vaut `active`, les lots actifs sont servis, `roll_series_end` est la dernière séance ;
- `/app/.env` est absent de la nouvelle image.

Étapes 0 à 2 faites le 27/10/2026 :

- **Git** : VPS sur `2b177a0`, sauvegarde `update_20261027_024711_bcab53e.tgz`.
- **Image** : `08f605375729`, sans `/app/.env`. L'ancienne est gardée comme `prediction-api:pre-lotc-20261027`.
- **`.env`** : ligne `PREDICTION_JOURNAL` ajoutée, ancien fichier dans `/opt/prediction-backups/env_before_journal_20261027`.
- **API recréée** avec `--no-deps` :
  - `/health` répond 200 en 30 s, en mode `active` sur les lots actifs ;
  - `roll_series_end` vaut 21/10 ;
  - `./data` et `./scripts` sont montés ;
  - le journal public est lu et écrit dans `prediction_journal_public.jsonl`.

### 3. Décision de l'écrivain

Option A ou B, ci-dessus.

### 4. Si A : amorçage PC vers VPS, une fois

1. Lancer un dernier passage sur le PC.
2. Copier vers le VPS :
   - `data/` : journal, `cocoa_news_event_versions.jsonl`, `cocoa_weather_*.csv`, `databento_cocoa_calendar_raw.csv`, `cocoa_oi_receipts.csv`, CSV historiques, `cafe/`, `freeze/`, `status/`, `quarantine/` (environ 8 Mo) ;
   - `models/candidates/<marché>/` : lot en place, lot précédent et `TRAINED_LOT.json` (environ 490 Mo) ;
   - `config/experimental_release_*.json` et `previous_*`.
3. Commiter `pipeline.journal_writer: vps`.
4. Faire `git pull` sur le PC, puis lancer `update.sh` sur le VPS.

Option A retenue. Copie faite le 27/10/2026 :

- **Volume** : 171 fichiers (118 Mo), SHA-256 vérifiés 171/171.
- **Copiés** : `data/`, sauf la série sans saut, identique des deux côtés. Le journal expérimental fait 2 360 lignes.
- **Lots en place** :
  - cacao : `20261020_055547` et `20261020_052744` ;
  - robusta : `20261013_045834` et `20261010_093542`, avec leurs N-HiTS.
- **Manifestes** : les deux manifestes cacao du PC.
- **Fichiers remplacés** : les 3 fichiers du VPS remplacés sont dans `/opt/prediction-backups/seed_20261027/`.
- **À noter** : il n'y a pas eu de dernier passage sur le PC, car ses données s'arrêtent au 22/10. Le premier passage du VPS rattrapera les séances suivantes.

### 5. Essais

Lancer `scripts/vps/pipeline.sh --dry-run`, puis un passage réel à la main. Lire `logs/pipeline/`.

### 6. Cron

Commande d'installation plus haut.

### 7. Alertes

Le propriétaire ajoute le canal d'alerte dans le `.env` du VPS, par exemple une ligne
`ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<jeton>`. Le cron relit `.env` à chaque
passage : rien à redémarrer. Vérifier ensuite avec `scripts/vps/pipeline.sh --test-alert` (deux
messages attendus). L'API publique ne voit la ligne qu'à sa prochaine recréation.

## Après la bascule

- **Code.** Il passe par git : commit sur `main`, puis `scripts/vps/update.sh <sha>`, puis `up -d --no-deps --force-recreate api` si le code servi change.
  - `src/` et `config/` sont montés dans le conteneur : quand seuls eux changent, compose ne voit rien et `up -d --no-deps api` répond `Running`. L'API garde alors l'ancien code en mémoire (constaté le 27/10/2026).
  - `--force-recreate` recrée le conteneur ; `--no-deps` laisse redis et mlflow tels quels.
- **Fichiers gardés.** `update.sh` garde ce que le traitement du VPS réécrit : les manifestes expérimentaux et les `reports/walk_forward/**/freeze.json` du walk-forward. Il refuse (code 3) si un autre fichier contient un contenu absent de git.
- **Promotion.** Elle reste manuelle : un commit de `config/active_release_*.json` qui nomme un lot présent sur le VPS, puis `update.sh` et `up -d --no-deps --force-recreate api` : le manifeste actif n'est lu qu'au démarrage.
- **`deploy_models.ps1`.** Il n'est plus utilisé : il recopierait du code CRLF et des modèles du PC.
