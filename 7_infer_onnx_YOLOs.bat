@echo off
REM ============================================================
REM  Phase 7 (YOLO11s): run best.onnx and draw predicted circles.
REM  onnxruntime consumer (no torch/ultralytics). Export had no NMS,
REM  so the script does confidence threshold + NMS itself.
REM
REM  MODEL: absolute path to your exported yolo11s model (train_s).
REM  By default it sits in the run's weights folder; edit this if your
REM  run is train_s2/... or the model lives on another drive. You can
REM  also pass the path as the first argument:
REM     7_infer_onnx_YOLOs.bat "D:\path\to\best.onnx"
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set "MODEL=%~dp0runs\detect\train_s\weights\best.onnx"
if not "%~1"=="" set "MODEL=%~1"

python tools\infer_onnx.py "%~dp0jsonOutputs" --model "%MODEL%" --imgsz 640 --viz "%~dp0viz_infer_s" --out "%~dp0infer_out_s"
pause
