@echo off
REM ============================================================
REM  Phase 4 (resume): continue an interrupted training run.
REM  Picks up from runs\detect\train\weights\last.pt (saved at the
REM  end of each epoch). Use this INSTEAD of 4_train.bat after a
REM  crash / closed console -- re-running 4_train.bat starts over.
REM  resume=True ignores all other args; it reuses args.yaml + the
REM  same dataset\data.yaml split (do NOT re-run 3_make_split.bat
REM  in between -- it unfreezes the test block).
REM  The model path is pinned to this project folder via %~dp0.
REM  Adjust the run folder (train, train2, ...) if needed.
REM ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
yolo detect train resume model="%~dp0runs\detect\train\weights\last.pt"
pause
