@echo off
REM Script de configuration des tâches planifiées Windows
REM Exécuter en tant qu'administrateur

echo ================================================================================
echo CONFIGURATION DES TACHES PLANIFIEES
echo ================================================================================

REM Obtenir le répertoire du projet
set PROJECT_DIR=%~dp0..
cd /d "%PROJECT_DIR%"

echo.
echo Repertoire du projet: %PROJECT_DIR%
echo.

REM ============================================================================
REM 1. COLLECTE AUTOMATIQUE DES DONNEES (Tous les jours à 18h)
REM ============================================================================
echo [1/2] Configuration de la collecte automatique...

schtasks /Create /TN "CocoaPrice_DataCollection" /TR "python %PROJECT_DIR%\scripts\auto_collect.py" /SC DAILY /ST 18:00 /F

if %ERRORLEVEL% EQU 0 (
    echo    ✓ Collecte automatique configuree: Tous les jours a 18h00
) else (
    echo    ✗ Erreur configuration collecte
)

REM ============================================================================
REM 2. REENTRAINEMENT AUTOMATIQUE (Tous les lundis à 02h)
REM ============================================================================
echo.
echo [2/2] Configuration du reentrainement automatique...

schtasks /Create /TN "CocoaPrice_AutoRetrain" /TR "python %PROJECT_DIR%\scripts\auto_retrain.py" /SC WEEKLY /D MON /ST 02:00 /F

if %ERRORLEVEL% EQU 0 (
    echo    ✓ Reentrainement automatique configure: Tous les lundis a 02h00
) else (
    echo    ✗ Erreur configuration reentrainement
)

echo.
echo ================================================================================
echo RESUME DES TACHES PLANIFIEES
echo ================================================================================
echo.

schtasks /Query /TN "CocoaPrice_DataCollection" /FO LIST
echo.
schtasks /Query /TN "CocoaPrice_AutoRetrain" /FO LIST

echo.
echo ================================================================================
echo ✓ CONFIGURATION TERMINEE
echo ================================================================================
echo.
echo Pour modifier les horaires:
echo   - Ouvrir "Planificateur de taches" (taskschd.msc)
echo   - Chercher "CocoaPrice_DataCollection" et "CocoaPrice_AutoRetrain"
echo.
echo Pour desactiver:
echo   schtasks /Change /TN "CocoaPrice_DataCollection" /DISABLE
echo   schtasks /Change /TN "CocoaPrice_AutoRetrain" /DISABLE
echo.
pause
