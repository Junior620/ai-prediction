@echo off
setlocal
REM ================================================================================
REM MISE A JOUR — enveloppe de scripts\run_pipeline.py
REM ================================================================================
REM Chaque jour : collecte, gel jusqu'a la derniere journee complete, decision de
REM reentrainement, previsions journalisees avec le lot en place (J+1 a J+30, J+60, J+90
REM au journal seulement), API, evaluation, deploiement des donnees seules.
REM Le samedi, un lot de plus de 7 jours, --retrain, un choc de marche ou une degradation
REM declenchent l'entrainement. Jamais sur une seance incomplete. Aucun lot n'est promu.
REM
REM Usage : update_system.bat [--unattended] [--mode weekly] [--retrain] [--dry-run] [--preview-journal]
REM   --unattended : aucune pause (tache planifiee).
REM Codes de sortie (ceux de run_pipeline.py) :
REM   0 succes complet
REM   1 echec reel (collecte cacao, instantane, serie sans saut, entrainement, journal, manifeste)
REM   2 source indisponible, base en retard ou seance incomplete
REM   3 quarantaine (prix sans date de seance ou jour ferme, rien d'ecrit)
REM   4 succes partiel (API non confirmee, robusta non mis a jour, VPS non deploye)
REM ================================================================================

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set UNATTENDED=0
for %%a in (%*) do if /i "%%~a"=="--unattended" set UNATTENDED=1

call venv_py311\Scripts\python.exe scripts\run_pipeline.py %*
set FINAL=%ERRORLEVEL%

if %UNATTENDED% equ 0 pause
exit /b %FINAL%
