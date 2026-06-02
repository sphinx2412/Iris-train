@echo off
REM ============================================================
REM  Phase 0: create .venv (if missing) + install dependencies into it.
REM  Run once. CUDA torch line is for the training box (RTX 3060).
REM  On a data-prep-only machine you can skip the torch line.
REM ============================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtualenv in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create .venv ^(is python on PATH?^).
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing CUDA torch (training box only; comment out if CPU-only data prep)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo Installing the rest...
pip install -r requirements.txt
echo.
echo Done.
pause
