@echo off
setlocal enabledelayedexpansion
REM ================================================================================
REM MISE A JOUR COMPLETE — CACAO + CAFE ROBUSTA
REM ================================================================================
REM 1. Collecte prix cacao (ICE London) + robusta (Investing.com)
REM 2. Collecte news + sentiment (cacao)
REM 3. Reentrainement hybride + N-HiTS pour les deux marches
REM 4. Redemarrage API locale (decouverte auto des modeles)
REM 5. Verification predictions API locale
REM 6. Deploy modeles vers VPS Contabo + restart API distante
REM ================================================================================

set PYTHONIOENCODING=utf-8

REM Charger API_TOKEN depuis .env (pas de JWT en dur dans le repo)
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
echo   MISE A JOUR COMPLETE — CACAO + CAFE ROBUSTA
echo ================================================================================
echo.

REM Verifier Docker
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
REM ETAPE 1: COLLECTE DES PRIX
REM ================================================================================
echo ================================================================================
echo ETAPE 1/6: COLLECTE DES PRIX
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

echo [INFO] Cafe robusta — Investing.com (RCU6)...
call venv_py311\Scripts\python.exe collect_coffee_robusta_price.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Echec collecte robusta — non bloquant
) else (
    echo [OK] Prix robusta collecte
)
echo.

REM ================================================================================
REM ETAPE 2: NEWS ET SENTIMENT
REM ================================================================================
echo ================================================================================
echo ETAPE 2/6: COLLECTE DES NEWS ET ANALYSE DU SENTIMENT
echo ================================================================================
echo.

echo [INFO] Collecte des articles de news et analyse du sentiment...
call venv_py311\Scripts\python.exe collect_news.py
if errorlevel 1 (
    echo [ERREUR] Echec de la collecte des news
    pause
    exit /b 1
)
echo [OK] News collectees et sentiment analyse
echo.

REM ================================================================================
REM ETAPE 3: REENTRAINEMENT DES MODELES (CACAO + ROBUSTA)
REM ================================================================================
echo ================================================================================
echo ETAPE 3/6: REENTRAINEMENT DES MODELES
echo ================================================================================
echo.

echo --- CACAO (Prophet + XGBoost) ---
call venv_py311\Scripts\python.exe train_hybrid_improved.py --market cocoa
if errorlevel 1 (
    echo [ERREUR] Echec reentrainement hybride cacao
    pause
    exit /b 1
)
echo [OK] Hybride cacao reentraine
echo.

echo --- CACAO WALK-FORWARD (metriques dashboard, origines recentes) ---
call venv_py311\Scripts\python.exe scripts\run_walk_forward_validation.py --market cocoa --max-origins 24 --skip-nhits --horizons 1 7 14 30 --origin-window recent --direct-hstep --skip-calibration --output-dir reports/walk_forward/daily
if errorlevel 1 (
    echo [AVERTISSEMENT] Walk-forward cacao echoue — section Performance peut rester vide
) else (
    echo [OK] Walk-forward cacao genere
)
echo.
echo --- CACAO WALK-FORWARD HISTORIQUE (rapport distinct, ne remplace pas le tableau de bord) ---
call venv_py311\Scripts\python.exe scripts\run_walk_forward_validation.py --market cocoa --max-origins 48 --skip-nhits --horizons 1 7 14 30 --origin-window historical --skip-calibration --output-dir reports/walk_forward/historical
if errorlevel 1 (
    echo [AVERTISSEMENT] Walk-forward historique cacao echoue
) else (
    echo [OK] Walk-forward historique cacao genere
)
echo.

echo --- CAFE ROBUSTA (Prophet + XGBoost) ---
call venv_py311\Scripts\python.exe train_hybrid_improved.py --market coffee_robusta
if errorlevel 1 (
    echo [AVERTISSEMENT] Echec reentrainement hybride robusta — non bloquant
    echo                Relancer: update_system_resume.bat
) else (
    echo [OK] Hybride robusta reentraine
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

echo --- CACAO (N-HiTS via Docker Linux) ---
echo [INFO] Contourne le blocage Windows Ray — 2-5 min...
docker compose exec -T api python -u train_nhits.py --market cocoa
if errorlevel 1 (
    echo [AVERTISSEMENT] N-HiTS cacao non entraine — fallback: docker compose run
    docker compose run --rm --no-deps api python -u train_nhits.py --market cocoa
    if errorlevel 1 (
        echo [AVERTISSEMENT] N-HiTS cacao non entraine — 2 moteurs pour le cacao
    ) else (
        echo [OK] N-HiTS cacao entraine
    )
) else (
    echo [OK] N-HiTS cacao entraine
)
echo.

echo --- CAFE ROBUSTA (N-HiTS via Docker Linux) ---
echo [INFO] Contourne le blocage Windows Ray — 2-5 min...
docker compose exec -T api python -u train_nhits.py --market coffee_robusta
if errorlevel 1 (
    echo [AVERTISSEMENT] N-HiTS robusta non entraine — fallback: docker compose run
    docker compose run --rm --no-deps api python -u train_nhits.py --market coffee_robusta
    if errorlevel 1 (
        echo [AVERTISSEMENT] N-HiTS robusta non entraine — 2 moteurs pour le robusta
    ) else (
        echo [OK] N-HiTS robusta entraine
    )
) else (
    echo [OK] N-HiTS robusta entraine
)
echo.

echo --- CACAO COURBE A TERME Londres (XGBoost ranks C.v.0..3) ---
call venv_py311\Scripts\python.exe train_futures_curve.py --source london
if errorlevel 1 (
    echo [AVERTISSEMENT] Modeles futures Londres non entraines — fallback Investing
    call venv_py311\Scripts\python.exe train_futures_curve.py --source investing
) else (
    echo [OK] Courbe a terme Londres entrainee
)
echo.

echo --- CACAO COURBE A TERME Londres mois nommes (DEC26…) ---
call venv_py311\Scripts\python.exe train_futures_curve.py --source london_named
if errorlevel 1 (
    echo [AVERTISSEMENT] Modeles DEC26… non entraines — fallback spot_shift
) else (
    echo [OK] Courbe Londres nommee entrainee
)
echo.

echo [INFO] Nettoyage des anciens modeles ^(garde 5 derniers^)...
call venv_py311\Scripts\python.exe scripts\cleanup_old_models.py --keep 5

echo [INFO] Evaluation predictions vs prix reels...
call venv_py311\Scripts\python.exe evaluate_predictions.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Evaluation precision echouee — non bloquant
) else (
    echo [OK] Metriques de precision mises a jour
)
echo.

REM ================================================================================
REM ETAPE 4: REDEMARRAGE API
REM ================================================================================
echo ================================================================================
echo ETAPE 4/6: REDEMARRAGE DE L'API
echo ================================================================================
echo.

echo [INFO] L'API charge automatiquement les derniers modeles dans:
echo        - models\              (cacao)
echo        - models\coffee_robusta\ (robusta)
echo.

echo [INFO] Redemarrage de l'API Docker...
docker-compose restart api

echo [INFO] Attente du demarrage de l'API (jusqu'a 2 minutes)...
set API_READY=0
for /L %%i in (1,1,24) do (
    if !API_READY! equ 0 (
        timeout /t 5 /nobreak >nul
        powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/docs' -UseBasicParsing -TimeoutSec 3; exit 0 } catch { exit 1 }" >nul 2>&1
        if !errorlevel! equ 0 (
            set API_READY=1
            echo [OK] API prete!
        ) else (
            echo [INFO] API pas encore prete... (%%i/24)
        )
    )
)

if !API_READY! equ 0 (
    echo [AVERTISSEMENT] API lente. Logs: docker logs cocoa-api --tail 30
) else (
    echo [OK] API redemarree
)
echo.

REM ================================================================================
REM ETAPE 5: VERIFICATION
REM ================================================================================
echo ================================================================================
echo ETAPE 5/6: VERIFICATION DES PREDICTIONS
echo ================================================================================
echo.

echo [INFO] Test prediction CACAO (ICE London, GBP/T)...
powershell -NoProfile -Command "$t=$env:API_TOKEN; $headers = @{'Authorization' = \"Bearer $t\"; 'Content-Type' = 'application/json'}; $body = @{market = 'ICE_NY'; horizons = @(1); include_sentiment = $true} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/predict' -Method Post -Headers $headers -Body $body -TimeoutSec 60; Write-Host ('  Prix: ' + [math]::Round($r.current_price,2) + ' GBP/T  J+1: ' + [math]::Round($r.predictions[0].price,2) + ' GBP/T') } catch { Write-Host ('  [AVERTISSEMENT] ' + $_.Exception.Message) }"

echo.
echo [INFO] Test prediction CAFE ROBUSTA (COFFEE_ROBUSTA)...
powershell -NoProfile -Command "$t=$env:API_TOKEN; $headers = @{'Authorization' = \"Bearer $t\"; 'Content-Type' = 'application/json'}; $body = @{market = 'COFFEE_ROBUSTA'; horizons = @(1); include_sentiment = $false} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/predict' -Method Post -Headers $headers -Body $body -TimeoutSec 60; Write-Host ('  Prix: $' + $r.current_price + ' USD/T  J+1: $' + $r.predictions[0].price) } catch { Write-Host ('  [AVERTISSEMENT] ' + $_.Exception.Message) }"

echo.
echo [INFO] Marches disponibles: GET http://localhost:8000/api/v1/markets
echo [INFO] Dashboard: http://localhost:3000 (cacao)  /  http://localhost:3000/coffee (robusta)
echo.

REM ================================================================================
REM ETAPE 6: DEPLOY VERS VPS
REM ================================================================================
echo ================================================================================
echo ETAPE 6/6: DEPLOY MODELES VERS VPS
echo ================================================================================
echo.

echo [INFO] Copie du manifeste actif seulement. Un candidat non accepte n'est pas promu.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_models.ps1"
if errorlevel 1 (
    echo [AVERTISSEMENT] Deploy VPS echoue — l'API locale est a jour, le VPS non.
    echo                Verifie SSH / .env.deploy puis lance: deploy_models.bat
) else (
    echo [OK] VPS mis a jour
)
echo.

REM ================================================================================
REM RESUME
REM ================================================================================
echo ================================================================================
echo   MISE A JOUR TERMINEE
echo ================================================================================
echo.
echo [OK] Prix cacao + robusta collectes
echo [OK] News et sentiment
echo [OK] Modeles hybrides cacao + robusta reentraines
echo [OK] N-HiTS cacao + robusta (si succes ci-dessus)
echo [OK] API locale redemarree
echo [OK] Deploy VPS (si SSH OK)
echo.
echo Prod: https://api.market.ste-scpb.com/health
echo Duree totale estimee: 10-20 minutes (selon N-HiTS)
echo ================================================================================
pause

