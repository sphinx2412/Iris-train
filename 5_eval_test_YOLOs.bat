@echo off
REM ============================================================
REM  Phase 5 (YOLO11s): honest eval on the FROZEN test split.
REM  Per-class precision/recall + mAP (pupil expected weaker).
REM  Reads the yolo11s run (train_s) instead of the yolo11m run.
REM  Paths pinned to this project folder via %~dp0 (independent of
REM  Ultralytics' global runs_dir). If your run is train_s2/...,
REM  change "train_s" in the model path below to match.
REM  name=val_s keeps the eval output separate from the m run's val.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect val model="%~dp0runs\detect\train_s\weights\best.pt" data="%~dp0dataset\data.yaml" imgsz=640 split=test project="%~dp0runs\detect" name=val_s
pause
