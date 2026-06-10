@echo off
REM ============================================================
REM  Phase 4 (YOLO11s): lighter model for fast phone inference.
REM  Same imgsz=1280 as the yolo11m run, but yolo11s is much
REM  lighter -> batch=-1 lets Ultralytics auto-fit VRAM (safe on
REM  3060 12 GB and 2070 8 GB). Tiny pupil (~14 px box @1280) stays
REM  borderline; if pupil recall is poor, raise imgsz to 1536.
REM
REM  name=train_s pins output to runs\detect\train_s INSIDE this
REM  project folder (via %~dp0), SEPARATE from the yolo11m run
REM  (train), so the two models never clobber each other. Re-running
REM  creates train_s2 -- to continue use 4b_resume_train_YOLOs.bat,
REM  to restart clean delete/rename the old run folder first.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train model=yolo11s.pt data="%~dp0dataset\data.yaml" imgsz=640 epochs=100 batch=0.85 patience=20 device=0 project="%~dp0runs\detect" name=train_s
pause
