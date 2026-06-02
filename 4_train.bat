@echo off
REM ============================================================
REM  Phase 4: train YOLO11. Defaults target RTX 3060 12 GB.
REM  batch=8 fits 12 GB; batch=12 OOMs it (yolo11m@1280 needs ~15-18 GB).
REM  On RTX 2070 8 GB: lower batch further, or imgsz=1024, or batch=-1
REM  (auto). Alternative: model=yolo11s.pt imgsz=1536.
REM
REM  project/name pin output to runs\detect\train INSIDE this project
REM  folder (via %~dp0), so results ignore Ultralytics' global runs_dir
REM  setting and never land on the wrong drive. Re-running this creates
REM  train2 (name "train" is taken) -- to continue use 4b_resume_train.bat,
REM  to restart clean delete/rename the old run folder first.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train model=yolo11m.pt data="%~dp0dataset\data.yaml" imgsz=1280 epochs=100 batch=8 patience=20 device=0 project="%~dp0runs\detect" name=train
pause
