@echo off
REM Launch the Streamlit UI for Windows

echo Launching Information Retrieval System UI...
echo ==========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Run Streamlit
streamlit run src\ui\streamlit_app.py

pause
