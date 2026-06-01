@echo off
REM ============================================================
REM  Phase 3: block split by id (+ buffer) -> dataset\ + data.yaml
REM  train ~75% / val ~15% / test ~10%. test = last contiguous
REM  block, FROZEN. Don't re-run this after training starts.
REM  Tune with: --ratios 0.75 0.15 0.10  --buffer 4
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tools\make_split.py
pause
