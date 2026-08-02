@echo off
title ATRATUS: GPU ACCELERATED
color 0A
set PYTHONIOENCODING=utf-8
cls
cd /d "%~dp0"

REM Auto-detect Python: conda env > system python
set PY_PATH=python
where conda >nul 2>nul
if %errorlevel% equ 0 (
    call conda activate gtrade_gpu 2>nul
)
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ or run setup_gpu.bat
    pause
    exit
)

:menu
cls
echo =======================================================
echo    ATRATUS
echo =======================================================
echo.
echo  CORE                     ANALYTICS
echo    [1] Full Cycle           [N] News Analyzer
echo    [2] Dashboard            [D] News Digest
echo    [3] Predict (Radar)      [R] Regime Detector
echo    [WU] Web UI (FastAPI)    [C] Correlation Alert
echo                             [WL] Watchlist
echo                             [T] Optuna Tune
echo  DATA / TRAINING           [P] Paper Trading
echo    [4] Data Update
echo    [5] Train Models       REPORTS
echo    [5C] Train Chunked
echo    [6] Backtest             [M] Model Health
echo                             [E] Export Signals CSV
echo  WHAT-IF SIMULATOR          [L] Signal Log
echo    [W1] Top-5  90d equal    [H] HTML Report
echo    [W2] Top-10 90d equal    [Q] Equity Curve
echo    [W3] Top-5 180d equal  OTHER
echo    [W4] Top-5  90d Kelly    [B] DB Backup
echo    [W5] Custom assets       [I] Install/Repair
echo  SERVICES
echo    [7] Telegram Bot  [8] Scheduler  [9] DB Audit  [F] DB Fix  [0] EXIT
echo    [SG] Publish live signals to the site (Supabase)
echo  GENOME
echo    [AG] Adopt a genome   [AS] What is adopted   [AR] Revert adoption
echo    [ABC] Configure a genome A/B     [ABR] Run the configured A/B
echo  RESEARCH / MAINTENANCE
echo    [RS] Auto-research agent (own menu)   [LC] Daily loop cycle
echo    [RC] Recalibrate live probabilities   [TP] Fit the timing policy
echo.
echo =======================================================
set /p choice="Select: "

if "%choice%"=="1" goto full_run
if "%choice%"=="2" goto dashboard
if /i "%choice%"=="WU" goto webui
if "%choice%"=="3" goto predict
if "%choice%"=="4" goto data_only
if "%choice%"=="5" goto train_only
if "%choice%"=="6" goto backtest
if "%choice%"=="7" goto telegram_bot
if "%choice%"=="8" goto scheduler
if "%choice%"=="9" goto db_check
if /i "%choice%"=="N" goto news
if /i "%choice%"=="D" goto digest
if /i "%choice%"=="R" goto regime
if /i "%choice%"=="C" goto corr
if /i "%choice%"=="WL" goto watchlist
if /i "%choice%"=="W1" goto whatif_top5
if /i "%choice%"=="W2" goto whatif_top10
if /i "%choice%"=="W3" goto whatif_180
if /i "%choice%"=="W4" goto whatif_kelly
if /i "%choice%"=="W5" goto whatif_custom
if /i "%choice%"=="P" goto paper
if /i "%choice%"=="M" goto model_health
if /i "%choice%"=="E" goto export
if /i "%choice%"=="SG" goto push_signals
if /i "%choice%"=="5C" goto train_chunked
if /i "%choice%"=="AG" goto adopt_genome
if /i "%choice%"=="AS" goto adopt_show
if /i "%choice%"=="AR" goto adopt_revert
if /i "%choice%"=="RS" goto auto_research
if /i "%choice%"=="LC" goto loop_cycle
if /i "%choice%"=="RC" goto recalibrate
if /i "%choice%"=="TP" goto timing_policy
if /i "%choice%"=="ABC" goto ab_configure
if /i "%choice%"=="ABR" goto ab_run
if /i "%choice%"=="L" goto signal_log
if /i "%choice%"=="H" goto report
if /i "%choice%"=="F" goto db_fix
if /i "%choice%"=="B" goto backup
if /i "%choice%"=="I" goto install_fix
if /i "%choice%"=="Q" goto equity
if /i "%choice%"=="T" goto optuna
if "%choice%"=="0" exit
goto menu

:full_run
cls
python data_engine.py
python train_hybrid.py
python -m streamlit run app.py
pause
goto menu

:dashboard
cls
python -m streamlit run app.py
pause
goto menu

:webui
cls
echo [Web UI] FastAPI on http://127.0.0.1:8000  (Ctrl+C to stop)
start "" http://127.0.0.1:8000
python -m uvicorn webapp:app --port 8000
pause
goto menu

:predict
cls
python predict.py
pause
goto menu

:data_only
cls
python data_engine.py
pause
goto menu

:train_only
cls
python train_hybrid.py
pause
goto menu

:train_chunked
cls
echo Chunked trainer: one fresh process per chunk, resumable, and a champion
echo changes only if the new model beats it. Add --force-promote only to rebuild
echo a baseline or repair registry metadata.
echo.
python train_chunked.py
pause
goto menu

:auto_research
cls
echo Hands over to auto_research.bat, which has its own menu (mode, proposer,
echo budget, objective, score basis). It never touches production: candidates
echo train into temp dirs and nothing is adopted without you.
echo.
REM  setlocal so the agent's GTRADE_* choices (chronos features, extra columns,
REM  objective) die with it. Without this they would leak into whatever the
REM  main menu runs next, and a later Train would silently pick them up.
setlocal
call auto_research.bat
endlocal
goto menu

:loop_cycle
cls
echo Daily pipeline (data, predict, reconcile) plus a drift scan. Proposals
echo appear on the /loop page; nothing retrains without your approval.
echo.
python loop_cycle.py
pause
goto menu

:recalibrate
cls
echo Refits the global live-calibration layer from verified outcomes and
echo REPLACES models\live_calib_global.pkl. Weekly is enough. Delete that file
echo to roll back to the raw model probabilities.
echo.
python recalibrate_live.py
pause
goto menu

:timing_policy
cls
echo Fits the entry-timing policy from the live track record and writes
echo timing_policy.json. It only takes effect when GTRADE_TIMING_POLICY=1.
echo.
python train_timing.py
pause
goto menu

:ab_configure
cls
python ab_build.py
pause
goto menu

:ab_run
cls
echo This trains the holdout once per arm, roughly 8 to 11 hours each.
echo Do not start it while a retrain is running: they compete for RAM and cores.
echo Stop the scheduler and do not run data_engine while it works - new bars
echo mid-run make the arms measure different windows.
echo.
python ab_build.py --run
pause
goto menu

:adopt_genome
cls
python adopt_genome.py
pause
goto menu

:adopt_show
cls
python adopt_genome.py --show
pause
goto menu

:adopt_revert
cls
python adopt_genome.py --revert
pause
goto menu

:backtest
cls
python backtest.py
pause
goto menu

:telegram_bot
cls
echo [INFO] Starting bot... (Do not close this window!)
python alert_bot.py
echo.
echo [WARNING] Bot stopped. Check above for errors.
pause
goto menu

:push_signals
cls
echo [Site] Publishing the latest signals snapshot to Supabase for the landing.
echo        Needs SUPABASE_URL + SUPABASE_SERVICE_KEY in .env (service key = secret).
echo.
python push_signals.py
pause
goto menu

:scheduler
cls
python scheduler.py
pause
goto menu

:db_check
cls
python db_check.py
pause
goto menu

:news
cls
python news_analyzer.py
pause
goto menu

:digest
cls
python news_analyzer.py --digest
pause
goto menu

:regime
cls
python regime_detector.py
pause
goto menu

:corr
cls
python correlation_alert.py
pause
goto menu

:watchlist
cls
python watchlist.py
pause
goto menu

:paper
cls
python paper_trading.py
pause
goto menu

:model_health
cls
python model_health.py
pause
goto menu

:export
cls
python export_signals.py
pause
goto menu

:signal_log
cls
python signal_log.py
pause
goto menu

:report
cls
python performance_report.py
pause
goto menu

:equity
cls
python equity_curve.py
pause
goto menu

:db_fix
cls
python db_check.py --fix
pause
goto menu

:backup
cls
python db_backup.py
pause
goto menu

:install_fix
cls
python -m pip install apimoex requests yfinance pandas "numpy<2" plotly streamlit sqlalchemy catboost scikit-learn pyTelegramBotAPI pysocks python-dotenv tabulate tqdm optuna fastapi uvicorn jinja2 --no-cache-dir
pause
goto menu

:optuna
cls
python optuna_tune.py
pause
goto menu

:whatif_top5
cls
echo [What-If] Top-5 активов, 90 дней, равное распределение...
python whatif_simulator.py --top 5 --days 90 --strategy equal
pause
goto menu

:whatif_top10
cls
echo [What-If] Top-10 активов, 90 дней, равное распределение...
python whatif_simulator.py --top 10 --days 90 --strategy equal
pause
goto menu

:whatif_180
cls
echo [What-If] Top-5 активов, 180 дней, равное распределение...
python whatif_simulator.py --top 5 --days 180 --strategy equal
pause
goto menu

:whatif_kelly
cls
echo [What-If] Top-5 активов, 90 дней, Kelly-аллокация...
python whatif_simulator.py --top 5 --days 90 --strategy kelly
pause
goto menu

:whatif_custom
cls
set /p WI_ASSETS="Активы через пробел (BTC ETH NVDA ...): "
set /p WI_DAYS="Количество дней (Enter = 90): "
if "%WI_DAYS%"=="" set WI_DAYS=90
set /p WI_CAP="Капитал USD (Enter = 10000): "
if "%WI_CAP%"=="" set WI_CAP=10000
echo.
echo [What-If] Активы: %WI_ASSETS% | Дней: %WI_DAYS% | Капитал: $%WI_CAP%
python whatif_simulator.py %WI_ASSETS% --days %WI_DAYS% --capital %WI_CAP%
pause
goto menu
