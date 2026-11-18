@echo off
REM PathSnap API - run on Laptop 1. Double-click this file (or run it from CMD).
cd /d "%~dp0"
if "%PATHSNAP_EVAL_DATA%"=="" set "PATHSNAP_EVAL_DATA=%USERPROFILE%\evaluator_data"
echo Evaluation data folder: %PATHSNAP_EVAL_DATA%\evaluator_test
echo This laptop's IP address (type it into the UI as http://IP:8000):
ipconfig | findstr /c:"IPv4"
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
pause
