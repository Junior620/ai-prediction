@echo off
REM Script pour configurer la collecte automatique des news sur le cacao
REM Exécute la collecte tous les jours à 09h00 et 15h00 (heure de Douala)

echo ================================================================================
echo Configuration de la collecte automatique des news
echo ================================================================================
echo.

REM Obtenir le chemin absolu du projet
set PROJECT_DIR=%~dp0..
set PYTHON_PATH=%PROJECT_DIR%\venv\Scripts\python.exe
set SCRIPT_PATH=%PROJECT_DIR%\scripts\auto_collect_news.py

echo Chemin du projet: %PROJECT_DIR%
echo Chemin Python: %PYTHON_PATH%
echo Chemin script: %SCRIPT_PATH%
echo.

REM Vérifier que Python existe
if not exist "%PYTHON_PATH%" (
    echo ERREUR: Python introuvable dans le venv
    echo Veuillez d'abord creer l'environnement virtuel
    pause
    exit /b 1
)

echo [1/2] Creation de la tache planifiee pour la collecte de news...
echo.

REM Supprimer la tâche si elle existe déjà
schtasks /Query /TN "CocoaPrice_NewsCollection_Morning" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Suppression de l'ancienne tache du matin...
    schtasks /Delete /TN "CocoaPrice_NewsCollection_Morning" /F >nul 2>&1
)

schtasks /Query /TN "CocoaPrice_NewsCollection_Afternoon" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Suppression de l'ancienne tache de l'apres-midi...
    schtasks /Delete /TN "CocoaPrice_NewsCollection_Afternoon" /F >nul 2>&1
)

REM Créer la tâche pour 09h00 (matin)
echo Creation de la tache du matin (09h00)...
schtasks /Create /TN "CocoaPrice_NewsCollection_Morning" /TR "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\"" /SC DAILY /ST 09:00 /F

if %ERRORLEVEL% EQU 0 (
    echo [OK] Tache du matin creee avec succes
) else (
    echo [ERREUR] Echec de la creation de la tache du matin
)

REM Créer la tâche pour 15h00 (après-midi)
echo Creation de la tache de l'apres-midi (15h00)...
schtasks /Create /TN "CocoaPrice_NewsCollection_Afternoon" /TR "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\"" /SC DAILY /ST 15:00 /F

if %ERRORLEVEL% EQU 0 (
    echo [OK] Tache de l'apres-midi creee avec succes
) else (
    echo [ERREUR] Echec de la creation de la tache de l'apres-midi
)

echo.
echo [2/2] Test de la collecte de news...
echo.

REM Tester le script
"%PYTHON_PATH%" "%SCRIPT_PATH%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo CONFIGURATION TERMINEE AVEC SUCCES
    echo ================================================================================
    echo.
    echo Taches planifiees creees:
    echo   - CocoaPrice_NewsCollection_Morning   : Tous les jours a 09h00
    echo   - CocoaPrice_NewsCollection_Afternoon : Tous les jours a 15h00
    echo.
    echo Les news seront collectees automatiquement 2 fois par jour.
    echo Les logs sont dans: %PROJECT_DIR%\logs\
    echo.
    echo Pour voir les taches:
    echo   schtasks /Query /TN "CocoaPrice_NewsCollection_Morning"
    echo   schtasks /Query /TN "CocoaPrice_NewsCollection_Afternoon"
    echo.
    echo Pour supprimer les taches:
    echo   schtasks /Delete /TN "CocoaPrice_NewsCollection_Morning" /F
    echo   schtasks /Delete /TN "CocoaPrice_NewsCollection_Afternoon" /F
    echo.
) else (
    echo.
    echo ================================================================================
    echo ERREUR LORS DU TEST
    echo ================================================================================
    echo.
    echo Le script de collecte a rencontre une erreur.
    echo Verifiez les logs dans: %PROJECT_DIR%\logs\
    echo.
)

pause
