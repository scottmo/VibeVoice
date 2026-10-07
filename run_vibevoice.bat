@echo off
setlocal

REM Anchor paths to this launcher so it works from any current directory.
pushd "%~dp0"

echo Starting VibeVoice from %CD%...

if not exist "venv\Scripts\python.exe" (
    echo Error: project virtual environment not found.
    echo Create it with Python 3.11 and install the project dependencies first.
    popd
    pause
    exit /b 1
)

if not exist ".env" (
    echo Warning: .env file not found. Copy .env-sample to .env to configure model loading.
)

echo Launching VibeVoice...
echo The local interface is available at http://localhost:7590
echo.

venv\Scripts\python.exe main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo VibeVoice has stopped with exit code %EXIT_CODE%.
popd
pause
exit /b %EXIT_CODE%
