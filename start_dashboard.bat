@echo off
echo ========================================
echo   DEMARRAGE DU DASHBOARD CACAO
echo ========================================
echo.

REM Activer l'environnement virtuel
call venv_py311\Scripts\activate.bat

REM Installer les dépendances si nécessaire
echo Installation des dependances...
pip install streamlit plotly --quiet

echo.
echo ========================================
echo   Dashboard demarre sur:
echo   http://localhost:8501
echo ========================================
echo.
echo Appuyez sur Ctrl+C pour arreter
echo.

REM Démarrer Streamlit
streamlit run dashboard.py
