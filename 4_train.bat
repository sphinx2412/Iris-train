@echo off
REM ============================================================
REM  Phase 4: train YOLO11. Defaults target RTX 3070 12 GB.
REM  If you train on RTX 2070 8 GB: lower batch, or imgsz=1024,
REM  or batch=-1 (auto). Alternative: model=yolo11s.pt imgsz=1536.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train model=yolo11m.pt data=dataset\data.yaml imgsz=1280 epochs=100 batch=12 patience=20 device=0
pause
