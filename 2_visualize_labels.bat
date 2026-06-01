@echo off
REM ============================================================
REM  Phase 2: sanity check. Draws 5 random labels back onto the
REM  photos into viz\ so you can compare with _debug_*.jpg.
REM  (Optional but recommended after every big annotation batch.)
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tools\visualize_labels.py --n 5
echo.
echo Check the viz\ folder.
pause
