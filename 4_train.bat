@echo off
REM ============================================================
REM  Phase 4: train YOLO11. Defaults target RTX 3060 12 GB.
REM  batch=8 fits 12 GB; batch=12 OOMs it (yolo11m@1280 needs ~15-18 GB).
REM  On RTX 2070 8 GB: lower batch further, or imgsz=1024, or batch=-1
REM  (auto). Alternative: model=yolo11s.pt imgsz=1536.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train model=yolo11m.pt data=dataset\data.yaml imgsz=1280 epochs=100 batch=8 patience=20 device=0
pause
