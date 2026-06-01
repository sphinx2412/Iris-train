@echo off
REM ============================================================
REM  Phase 1: convert jsonOutputs\*.json -> YOLO labels\*.txt
REM  Empty .txt is written for "no eye" (success:false) frames.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tools\json_to_yolo.py
pause
