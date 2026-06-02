@echo off
REM ============================================================
REM  Phase 5: honest eval on the FROZEN test split.
REM  Per-class precision/recall + mAP (pupil expected weaker).
REM  Paths pinned to this project folder via %~dp0 (independent of
REM  Ultralytics' global runs_dir). If your run is train2/train3/...,
REM  change "train" in the model path below to match.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect val model="%~dp0runs\detect\train\weights\best.pt" data="%~dp0dataset\data.yaml" split=test project="%~dp0runs\detect" name=val
pause
