@echo off
REM ============================================================
REM  Phase 4 (YOLO11s, resume): continue an interrupted yolo11s run.
REM  Picks up from runs\detect\train_s\weights\last.pt (saved at the
REM  end of each epoch). Use this INSTEAD of 4_train_YOLOs.bat after
REM  a crash / closed console -- re-running 4_train_YOLOs.bat starts
REM  over. resume=True ignores all other args; it reuses args.yaml +
REM  the same dataset\data.yaml split (do NOT re-run 3_make_split.bat
REM  in between -- it unfreezes the test block).
REM  The model path is pinned to this project folder via %~dp0.
REM  Adjust the run folder (train_s, train_s2, ...) if needed.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train resume model="%~dp0runs\detect\train_s\weights\last.pt"
pause
