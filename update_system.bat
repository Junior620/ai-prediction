@echo off
setlocal enabledelayedexpansion
REM ================================================================================
REM MISE A JOUR — CACAO D'ABORD, PUIS ROBUSTA
REM ================================================================================
REM 1. Collecte et fige le cacao. La collecte ne change plus ensuite.
REM 2. Entraine le lot cacao nomme, horizons 1 a 30. Pas de N-HiTS cacao.
REM 3. Journalise les previsions et le diagnostic ancien/nouveau.
REM 4. Charge ce lot experimental et verifie l'API avant le robusta.
REM 5. Repete la sequence pour le robusta, avec N-HiTS.
REM 6. Diagnostics secondaires. Les etudes historiques longues restent hors de cette commande.
REM 7. Transfere le manifeste actif vers le VPS. Le lot experimental du soir reste local.
REM ================================================================================

set PYTHONIOENCODING=utf-8

if not defined API_TOKEN (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (`findstr /b /c:"API_TOKEN=" "%~dp0.env" 2^>nul`) do set "API_TOKEN=%%b"
)
if not defined API_TOKEN (
    echo [ERREUR] API_TOKEN absent. Ajoutez API_TOKEN=... dans .env ^(generate_jwt_token.py^).
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   MISE A JOUR — CACAO D'ABORD, PUIS ROBUSTA
echo ================================================================================
echo.

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Docker n'est pas en cours d'execution.
    echo Veuillez demarrer Docker Desktop et reessayer.
    pause
    exit /b 1
)

echo [INFO] Docker est operationnel
echo.

REM ================================================================================
REM CACAO — collecte, gel, entrainement, journal, API
REM ================================================================================
echo ================================================================================
echo CACAO 1/4: COLLECTE
echo ================================================================================
echo.

echo [INFO] Cacao — ICE London (Playwright + fallbacks)...
call venv_py311\Scripts\python.exe collect_ice_london_cocoa.py
if errorlevel 1 (
    echo [ERREUR] Echec collecte cacao ICE London
    pause
    exit /b 1
)
echo [OK] Prix cacao ICE London collecte
echo.

echo [INFO] Contrats a terme Londres mois nommes (DEC26, MAR27...)...
call venv_py311\Scripts\python.exe collect_london_named_futures.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Courbe Londres nommee indisponible — non bloquant
) else (
    echo [OK] Courbe Londres nommee collectee
)
echo.

echo [INFO] Courbe a terme cacao — mois nommes NY Investing (Dec26, Mar27...)...
call venv_py311\Scripts\python.exe collect_cocoa_futures_investing.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Investing futures echoue — tentative Yahoo collect_futures.py
    call venv_py311\Scripts\python.exe collect_futures.py
    if errorlevel 1 (
        echo [AVERTISSEMENT] Courbe futures NY non mise a jour — non bloquant
    ) else (
        echo [OK] Courbe futures Yahoo collectee
    )
) else (
    echo [OK] Courbe futures Investing collectee
)
echo.

echo [INFO] Meteo regions cacao (WeatherAPI) — non bloquant...
call venv_py311\Scripts\python.exe collect_cocoa_weather.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Meteo non mise a jour — non bloquant
) else (
    echo [OK] Meteo cacao collectee
)
echo.

echo [INFO] Collecte des articles de news et analyse du sentiment...
call venv_py311\Scripts\python.exe collect_news.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Echec de la collecte des news — le calcul cacao continue
) else (
    echo [OK] News collectees et sentiment analyse
)
echo.

echo [INFO] Versions d'evenements NLP — non bloquant...
call venv_py311\Scripts\python.exe scripts\record_news_events.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Evenements NLP non enregistres — non bloquant
) else (
    echo [OK] Evenements NLP enregistres
)
echo.

echo [INFO] Gel des donnees cacao. La collecte ne sera plus relancee avant le journal.
call venv_py311\Scripts\python.exe scripts\freeze_market_snapshot.py --market cocoa --output data\freeze\cocoa_prices.csv
if errorlevel 1 (
    echo [ERREUR] Instantane cacao non fige
    pause
    exit /b 1
)

echo ================================================================================
echo CACAO 2/4: ENTRAINEMENT DU LOT NOMME
echo ================================================================================
echo.

call venv_py311\Scripts\python.exe train_hybrid_improved.py --market cocoa --snapshot data\freeze\cocoa_prices.csv
if errorlevel 1 (
    echo [ERREUR] Echec reentrainement cacao. Le nouveau modele n'est pas annonce.
    echo [INFO] Lot encore charge:
    call venv_py311\Scripts\python.exe -c "import json; from pathlib import Path; p=Path('config/experimental_release_cocoa.json'); print(json.loads(p.read_text(encoding='utf-8')).get('version') if p.exists() else 'aucun')"
    pause
    exit /b 1
)
echo [OK] Hybride cacao reentraine. N-HiTS cacao non entraine: poids nuls.
echo.

echo ================================================================================
echo CACAO 3/4: JOURNAL DES PREVISIONS
echo ================================================================================
echo.

call venv_py311\Scripts\python.exe scripts\journal_forecast_revisions.py --market cocoa --snapshot data\freeze\cocoa_prices.csv
if errorlevel 1 (
    echo [ERREUR] Journal cacao non ecrit. Relancer cette commande reprend le meme identifiant.
    echo [INFO] Le nouveau lot n'est pas annonce.
    pause
    exit /b 1
)

call venv_py311\Scripts\python.exe scripts\write_experimental_release.py --market cocoa
if errorlevel 1 (
    echo [ERREUR] Manifeste experimental cacao non ecrit. Le dernier dossier n'est pas utilise.
    pause
    exit /b 1
)

echo ================================================================================
echo CACAO 4/4: API LOCALE
echo ================================================================================
echo.

set PREDICTION_RELEASE_MODE=experimental
docker compose up -d --force-recreate api

call :wait_for_api

set COCOA_PARTIAL=0
if !API_READY! equ 1 (
    echo [INFO] Controle du lot cacao nomme, J+1 J+7 J+14 J+30...
    call venv_py311\Scripts\python.exe scripts\verify_served_forecast.py --market cocoa
        if errorlevel 3 (
        echo [AVERTISSEMENT] Cacao: succes partiel, au moins un horizon sans prix. Pas de succes complet.
        set COCOA_PARTIAL=1
    ) else if errorlevel 1 (
        echo [AVERTISSEMENT] Cacao partiel: le lot n'est pas confirme comme consultable.
        set COCOA_PARTIAL=1
    )
) else (
    echo [AVERTISSEMENT] API cacao non prete. Le lot n'est pas annonce comme consultable.
    set COCOA_PARTIAL=1
)
echo.

REM ================================================================================
REM ROBUSTA — un echec ici ne retire pas le cacao
REM ================================================================================
echo ================================================================================
echo ROBUSTA 1/4: COLLECTE ET GEL
echo ================================================================================
echo.

set ROBUSTA_OK=1
set ROBUSTA_COLLECT_TRY=0
:robusta_collect_retry
echo [INFO] Cafe robusta — Investing.com (RCU6)...
call venv_py311\Scripts\python.exe collect_coffee_robusta_price.py
if errorlevel 1 (
    set /a ROBUSTA_COLLECT_TRY+=1
    if !ROBUSTA_COLLECT_TRY! lss 3 (
        echo [AVERTISSEMENT] Collecte robusta echouee ^(!ROBUSTA_COLLECT_TRY!/3^). Nouvelle tentative dans 45 s...
        timeout /t 45 /nobreak >nul
        goto robusta_collect_retry
    )
    echo [AVERTISSEMENT] Echec collecte robusta apres 3 tentatives — le cacao reste consultable
    set ROBUSTA_OK=0
)
if !ROBUSTA_OK! equ 1 (
    call venv_py311\Scripts\python.exe scripts\freeze_market_snapshot.py --market coffee_robusta --output data\freeze\coffee_robusta_prices.csv
    if errorlevel 1 (
        echo [AVERTISSEMENT] Instantane robusta non fige
        set ROBUSTA_OK=0
    )
)

if !ROBUSTA_OK! equ 1 (
    echo ================================================================================
    echo ROBUSTA 2/4: ENTRAINEMENT
    echo ================================================================================
    call venv_py311\Scripts\python.exe train_hybrid_improved.py --market coffee_robusta --snapshot data\freeze\coffee_robusta_prices.csv
    if errorlevel 1 (
        echo [AVERTISSEMENT] Echec reentrainement robusta. Le nouveau modele n'est pas annonce.
        echo [INFO] Lot robusta encore charge:
        call venv_py311\Scripts\python.exe -c "import json; from pathlib import Path; p=Path('config/experimental_release_coffee_robusta.json'); print(json.loads(p.read_text(encoding='utf-8')).get('version') if p.exists() else 'aucun')"
        set ROBUSTA_OK=0
    )
)

if !ROBUSTA_OK! equ 1 (
    echo [INFO] N-HiTS robusta via Docker Linux...
    docker compose exec -T api python -u train_nhits.py --market coffee_robusta --snapshot data/freeze/coffee_robusta_prices.csv
    if errorlevel 1 (
        docker compose run --rm --no-deps api python -u train_nhits.py --market coffee_robusta --snapshot data/freeze/coffee_robusta_prices.csv
        if errorlevel 1 (
            echo [AVERTISSEMENT] N-HiTS robusta non entraine. Le nouveau modele n'est pas annonce.
            set ROBUSTA_OK=0
        )
    )
)

if !ROBUSTA_OK! equ 1 (
    echo ================================================================================
    echo ROBUSTA 3/4: JOURNAL
    echo ================================================================================
    docker compose run --rm --no-deps -v "%CD%\scripts:/app/scripts:ro" api python -u scripts/journal_forecast_revisions.py --market coffee_robusta --snapshot data/freeze/coffee_robusta_prices.csv
    if errorlevel 1 (
        echo [ERREUR] Journal robusta non ecrit. Relancer reprend le meme identifiant.
        set ROBUSTA_OK=0
    )
)

if !ROBUSTA_OK! equ 1 (
    call venv_py311\Scripts\python.exe scripts\write_experimental_release.py --market coffee_robusta
    if errorlevel 1 (
        echo [AVERTISSEMENT] Manifeste experimental robusta non ecrit
        set ROBUSTA_OK=0
    )
)

if !ROBUSTA_OK! equ 1 (
    echo ================================================================================
    echo ROBUSTA 4/4: API
    echo ================================================================================
    set PREDICTION_RELEASE_MODE=experimental
    docker compose up -d --force-recreate api
    call :wait_for_api
    if !API_READY! equ 1 (
        echo [INFO] Controle du lot robusta nomme, J+1 J+7 J+14 J+30...
        call venv_py311\Scripts\python.exe scripts\verify_served_forecast.py --market coffee_robusta
        if errorlevel 3 (
            echo [AVERTISSEMENT] Robusta: succes partiel, au moins un horizon sans prix. Le cacao reste en place.
            set ROBUSTA_OK=0
        ) else if errorlevel 1 (
            echo [AVERTISSEMENT] Robusta partiel: le lot n'est pas confirme comme consultable. Le cacao reste en place.
            set ROBUSTA_OK=0
        )
    ) else (
        echo [AVERTISSEMENT] API robusta non prete. Le cacao deja charge n'est pas retire.
        set ROBUSTA_OK=0
    )
) else (
    echo [AVERTISSEMENT] Robusta non mis a jour. Le cacao reste le lot charge si l'API a deja ete recreee.
)

echo.
echo ================================================================================
echo DIAGNOSTICS SECONDAIRES
echo ================================================================================
echo.

echo --- CACAO WALK-FORWARD (metriques dashboard, origines recentes) ---
call venv_py311\Scripts\python.exe scripts\run_walk_forward_validation.py --market cocoa --max-origins 24 --skip-nhits --horizons 1 7 14 30 --origin-window recent --direct-hstep --skip-calibration --output-dir reports/walk_forward/daily
if errorlevel 1 (
    echo [AVERTISSEMENT] Walk-forward cacao echoue — section Performance peut rester vide
) else (
    echo [OK] Walk-forward cacao genere
)
echo.

echo --- CAFE ROBUSTA WALK-FORWARD (metriques dashboard, origines recentes) ---
call venv_py311\Scripts\python.exe scripts\run_walk_forward_validation.py --market coffee_robusta --max-origins 24 --skip-nhits --horizons 1 7 14 30 --origin-window recent --direct-hstep --skip-calibration --output-dir reports/walk_forward/coffee_robusta/daily
if errorlevel 1 (
    echo [AVERTISSEMENT] Walk-forward robusta echoue — metriques cafe peuvent rester vides
) else (
    echo [OK] Walk-forward robusta genere
)
echo.

echo --- CACAO COURBE A TERME Londres ---
call venv_py311\Scripts\python.exe train_futures_curve.py --source london
if errorlevel 1 (
    echo [AVERTISSEMENT] Modeles futures Londres non entraines — fallback Investing
    call venv_py311\Scripts\python.exe train_futures_curve.py --source investing
) else (
    echo [OK] Courbe a terme Londres entrainee
)
echo.

echo --- CACAO COURBE A TERME Londres mois nommes ---
call venv_py311\Scripts\python.exe train_futures_curve.py --source london_named
if errorlevel 1 (
    echo [AVERTISSEMENT] Modeles DEC26… non entraines — fallback spot_shift
) else (
    echo [OK] Courbe Londres nommee entrainee
)
echo.

echo [INFO] Nettoyage des anciens modeles. Les lots des manifestes actif, experimental et precedent sont proteges.
call venv_py311\Scripts\python.exe scripts\cleanup_old_models.py --keep 5

echo [INFO] Evaluation predictions vs prix reels...
call venv_py311\Scripts\python.exe evaluate_predictions.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Evaluation precision echouee — non bloquant
) else (
    echo [OK] Metriques de precision mises a jour
)
echo.

echo [INFO] Marches disponibles: GET http://localhost:8000/api/v1/markets
echo [INFO] Suivi: GET http://localhost:8000/api/v1/forecast-tracking
echo [INFO] Dashboard: http://localhost:3000 (cacao)  /  http://localhost:3000/coffee (robusta)
echo.
echo ================================================================================
if !COCOA_PARTIAL! equ 1 (
    echo   MISE A JOUR PARTIELLE — le cacao n'est pas confirme comme consultable
) else if !ROBUSTA_OK! equ 0 (
    echo   MISE A JOUR PARTIELLE — cacao consulte, robusta non confirme
) else (
    echo   MISE A JOUR TERMINEE
)
echo ================================================================================
echo [INFO] Transfert du manifeste actif vers le VPS. Le lot experimental du soir reste local.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_models.ps1"
if errorlevel 1 (
    echo [AVERTISSEMENT] Deploy VPS echoue. L'API locale est a jour, le VPS non.
    echo                Verifie SSH / .env.deploy puis lance: deploy_models.bat
) else (
    echo [OK] VPS mis a jour
)
echo [INFO] Les etudes historiques longues ne font pas partie de cette commande.
echo ================================================================================
pause
goto :eof

:wait_for_api
echo [INFO] Attente du demarrage de l'API (jusqu'a 10 minutes apres recreate)...
echo [INFO] Premier demarrage apres recreate: telechargement FinBERT et chargement des modeles.
set API_READY=0
timeout /t 20 /nobreak >nul
for /L %%i in (1,1,60) do (
    if !API_READY! equ 0 (
        powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 10 | Out-Null; exit 0 } catch { if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 503) { exit 0 } else { exit 1 } }" >nul 2>&1
        if !errorlevel! equ 0 (
            set API_READY=1
            echo [OK] API demarree. /health repond.
        ) else (
            echo [INFO] API pas encore prete... (%%i/60^)
            timeout /t 10 /nobreak >nul
        )
    )
)
if !API_READY! equ 0 (
    echo [AVERTISSEMENT] API non prete apres 10 minutes. Consultez: docker logs cocoa-api --tail 80
)
exit /b 0
