@echo off
echo ========================================
echo Demarrage du Dashboard Premium Next.js
echo ========================================
echo.
echo Dashboard URL: http://localhost:3001
echo.
echo Appuyez sur Ctrl+C pour arreter
echo.

cd /d "%~dp0"
npm run dev -- -p 3001
