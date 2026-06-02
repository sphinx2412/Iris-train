@echo off
REM ============================================================
REM  Phase 3: block split by id (+ buffer) -> dataset\ + data.yaml
REM  train ~75% / val ~15% / test ~10%. test = last contiguous
REM  block, FROZEN. Don't re-run this after training starts.
REM  Tune via the RATIOS / BUFFER vars below.
REM  BUFFER=12: one person was shot <=12 times in a row, so dropping
REM  12 ids each side of a junction stops same-person leakage.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set RATIOS=0.75 0.15 0.10
set BUFFER=12
python tools\make_split.py --ratios %RATIOS% --buffer %BUFFER%
pause
