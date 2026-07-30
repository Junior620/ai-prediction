@echo off
REM Deploy les derniers modeles vers le VPS Contabo + restart API
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_models.ps1" %*
exit /b %ERRORLEVEL%
