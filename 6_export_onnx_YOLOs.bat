@echo off
REM ============================================================
REM  Phase 6 (YOLO11s): export best.pt -> ONNX (for onnxruntime).
REM  Reads the yolo11s run (train_s) instead of the yolo11m run.
REM  Model path pinned to this project folder via %~dp0; the .onnx is
REM  written next to best.pt. If your run is train_s2/..., change
REM  "train_s" in the path below to match.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo export model="%~dp0runs\detect\train_s\weights\best.pt" format=onnx imgsz=1280
pause
