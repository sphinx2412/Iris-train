@echo off
REM ============================================================
REM  Phase 6 (YOLO11s): export best.pt -> ONNX (for onnxruntime).
REM  Reads the yolo11s run (train_s) instead of the yolo11m run.
REM  Model path pinned to this project folder via %~dp0; the .onnx is
REM  written next to best.pt. If your run is train_s2/..., change
REM  "train_s" in the path below to match.
REM
REM  fp16: HALF is ON by default -> exports a float16 model. half=True for
REM  ONNX needs a CUDA GPU (device=0). Clear HALF below (set "HALF=") for a
REM  plain fp32 export.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set "HALF=1"
if /i "%~1"=="fp32" set "HALF="

set "OPTS=format=onnx imgsz=640"
if defined HALF set "OPTS=%OPTS% half=True device=0"

yolo export model="%~dp0runs\detect\train_s\weights\best.pt" %OPTS%
pause
