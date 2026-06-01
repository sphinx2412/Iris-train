@echo off
REM ============================================================
REM  Phase 6: export best.pt -> ONNX (for onnxruntime / later use).
REM  Adjust the run folder (train, train2, ...) if needed.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo export model=runs\detect\train\weights\best.pt format=onnx imgsz=1280
pause
