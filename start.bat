@echo off
:: Launch the PDF Chatbot using the project's virtual environment.
cd /d "%~dp0"

if not exist ".venv" (
    echo Virtual environment ^(.venv^) not found.
    echo Create it and install the dependencies first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate

echo Launching PDF Chatbot...
python run.py

:: Keep the window open if the app exited with an error
if %errorlevel% neq 0 (
    echo Application exited with an error.
    pause
)
