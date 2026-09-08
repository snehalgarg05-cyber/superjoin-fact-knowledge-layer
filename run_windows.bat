@echo off
:: Fact Knowledge Layer - Windows Start Script
:: Usage: Set your GROQ_API_KEY below and double-click this file

:: ── SET YOUR KEY HERE ──────────────────────────────────────
set GROQ_API_KEY=gsk_PASTE_YOUR_KEY_HERE
:: ──────────────────────────────────────────────────────────

if "%GROQ_API_KEY%"=="gsk_PASTE_YOUR_KEY_HERE" (
    echo ERROR: Please open run_windows.bat in Notepad and paste your Groq API key.
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Fact Knowledge Layer...
echo UI: http://localhost:8000/ui
echo.

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
