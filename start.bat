@echo off
:: Move to script directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment ^(.venv^) not found!
    echo Please create it first by running: python -m venv .venv --system-site-packages --without-pip
    pause
    exit /b 1
)

:: Activate venv
echo Activating virtual environment...
call .venv\Scripts\activate

:: Run the app
echo Launching PDF Chatbot...
python run.py

:: Pause on error/exit so the window doesn't close immediately
if %errorlevel% neq 0 (
    echo Application exited with an error.
    pause
)
