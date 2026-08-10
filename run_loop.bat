@echo off
REM Daily self-maintaining loop. Register with Windows Task Scheduler (run once):
REM   schtasks /Create /TN "Atratus Loop" /TR "\"%CD%\run_loop.bat\"" /SC DAILY /ST 23:30
REM Deploy this ONLY after the baseline training has finished.
cd /d "%~dp0"
REM NOTE: serving stays on the BASE env until the full retrain runs under
REM jackpot_gpu. TF 2.10 cannot open the current models/*.keras (Keras 3 zip),
REM so activating the GPU env here would drop every neural member from the
REM live ensemble. Add `call "%~dp0activate_env.bat"` once the retrain is done.
python loop_cycle.py >> loop.log 2>&1
