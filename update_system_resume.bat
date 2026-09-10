@echo off
setlocal enabledelayedexpansion
REM ================================================================================
REM REPRISE update_system — apres echec robusta (cacao deja OK)
REM Skip: collecte prix/news + train cacao + walk-forward
REM ================================================================================

set PYTHONIOENCODING=utf-8

if not defined API_TOKEN (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (`findstr /b /c:"API_TOKEN=" "%~dp0.env" 2^>nul`) do set "API_TOKEN=%%b"
)
if not defined API_TOKEN (
    echo [ERREUR] API_TOKEN absent dans .env
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Docker non demarre
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   REPRISE ENTRAINEMENT — ROBUSTA + N-HiTS + DEPLOY
echo ================================================================================
echo.

echo --- CAFE ROBUSTA (Prophet + XGBoost) ---
call venv_py311\Scripts\python.exe train_hybrid_improved.py --market coffee_robusta
if errorlevel 1 (
    echo [ERREUR] Echec reentrainement hybride robusta
    pause
    exit /b 1
)
echo [OK] Hybride robusta reentraine
echo.

echo --- CACAO (N-HiTS via Docker Linux) ---
docker compose exec -T api python -u train_nhits.py --market cocoa
if errorlevel 1 (
    echo [AVERTISSEMENT] N-HiTS cacao — fallback run
    docker compose run --rm --no-deps api python -u train_nhits.py --market cocoa
    if errorlevel 1 (
        echo [AVERTISSEMENT] N-HiTS cacao non entraine
    ) else (
        echo [OK] N-HiTS cacao entraine
    )
) else (
    echo [OK] N-HiTS cacao entraine
)
echo.

echo --- CAFE ROBUSTA (N-HiTS via Docker Linux) ---
docker compose exec -T api python -u train_nhits.py --market coffee_robusta
if errorlevel 1 (
    echo [AVERTISSEMENT] N-HiTS robusta — fallback run
    docker compose run --rm --no-deps api python -u train_nhits.py --market coffee_robusta
    if errorlevel 1 (
        echo [AVERTISSEMENT] N-HiTS robusta non entraine
    ) else (
        echo [OK] N-HiTS robusta entraine
    )
) else (
    echo [OK] N-HiTS robusta entraine
)
echo.

echo --- CACAO COURBE A TERME Londres ---
call venv_py311\Scripts\python.exe train_futures_curve.py --source london
if errorlevel 1 (
    echo [AVERTISSEMENT] Futures Londres — fallback Investing
    call venv_py311\Scripts\python.exe train_futures_curve.py --source investing
) else (
    echo [OK] Courbe a terme Londres entrainee
)
echo.

call venv_py311\Scripts\python.exe scripts\cleanup_old_models.py --keep 5
call venv_py311\Scripts\python.exe evaluate_predictions.py
if errorlevel 1 (
    echo [AVERTISSEMENT] Evaluation precision echouee — non bloquant
)

echo.
echo --- Redemarrage API ---
docker-compose restart api
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

echo.
echo --- Verification predictions ---
powershell -NoProfile -Command "$t=$env:API_TOKEN; $headers = @{'Authorization' = \"Bearer $t\"; 'Content-Type' = 'application/json'}; $body = @{market = 'ICE_NY'; horizons = @(1); include_sentiment = $true} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/predict' -Method Post -Headers $headers -Body $body -TimeoutSec 60; Write-Host ('  Cacao: ' + [math]::Round($r.current_price,2) + ' -> J+1 ' + [math]::Round($r.predictions[0].price,2)) } catch { Write-Host ('  [AVERTISSEMENT] ' + $_.Exception.Message) }"
powershell -NoProfile -Command "$t=$env:API_TOKEN; $headers = @{'Authorization' = \"Bearer $t\"; 'Content-Type' = 'application/json'}; $body = @{market = 'COFFEE_ROBUSTA'; horizons = @(1); include_sentiment = $false} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/predict' -Method Post -Headers $headers -Body $body -TimeoutSec 60; Write-Host ('  Robusta: $' + $r.current_price + ' -> J+1 $' + $r.predictions[0].price) } catch { Write-Host ('  [AVERTISSEMENT] ' + $_.Exception.Message) }"

echo.
echo --- Deploy VPS ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_models.ps1"
if errorlevel 1 (
    echo [AVERTISSEMENT] Deploy VPS echoue
) else (
    echo [OK] VPS mis a jour
)

echo.
echo ================================================================================
echo   REPRISE TERMINEE
echo ================================================================================
pause
