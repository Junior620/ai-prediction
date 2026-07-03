@echo off
echo ================================================================================
echo   Installation du Dashboard Next.js
echo ================================================================================
echo.

echo [1/3] Installation des dependances...
call npm install

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERREUR: L'installation a echoue
    pause
    exit /b 1
)

echo.
echo [2/3] Verification de la configuration...
if not exist .env.local (
    echo ERREUR: Fichier .env.local manquant
    pause
    exit /b 1
)

echo.
echo [3/3] Demarrage du serveur de developpement...
echo.
echo ================================================================================
echo   Dashboard demarre sur http://localhost:3000
echo   Appuyez sur Ctrl+C pour arreter
echo ================================================================================
echo.

call npm run dev

pause
