@echo off
REM ============================================================
REM  Phase 5: honest eval on the FROZEN test split.
REM  Per-class precision/recall + mAP (pupil expected weaker).
REM  Adjust the run folder (train, train2, ...) if needed.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect val model=runs\detect\train\weights\best.pt data=dataset\data.yaml split=test
pause
