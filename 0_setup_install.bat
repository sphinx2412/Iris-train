@echo off
REM ============================================================
REM  Phase 0: install dependencies into the local .venv
REM  Run once. CUDA torch line is for the training box (RTX 3070).
REM  On a data-prep-only machine you can skip the torch line.
REM ============================================================
cd /d "%~dp0"
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
