@echo off
REM ============================================================
REM  Phase 7: run best.onnx and draw predicted iris/pupil circles.
REM  onnxruntime consumer (no torch/ultralytics). Export had no NMS,
REM  so the script does confidence threshold + NMS itself.
REM
REM  MODEL: absolute path to your exported model. By default it sits in
REM  the run's weights folder; edit this if your run is train2/train3/...
REM  or the model lives on another drive. You can also pass the path as
REM  the first argument:   7_infer_onnx.bat "D:\path\to\best.onnx"
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set "MODEL=%~dp0runs\detect\train\weights\best.onnx"
if not "%~1"=="" set "MODEL=%~1"

python tools\infer_onnx.py "%~dp0jsonOutputs" --model "%MODEL%" --viz "%~dp0viz_infer" --out "%~dp0infer_out"
pause
