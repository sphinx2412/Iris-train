@echo off
REM ============================================================
REM  Phase 6: export best.pt -> ONNX (for onnxruntime / later use).
REM  Model path pinned to this project folder via %~dp0; the .onnx is
REM  written next to best.pt. If your run is train2/train3/..., change
REM  "train" in the path below to match.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo export model="%~dp0runs\detect\train\weights\best.pt" format=onnx imgsz=1280
pause
