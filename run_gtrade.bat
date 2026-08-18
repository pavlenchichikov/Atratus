@echo off
title ATRATUS: GPU ACCELERATED
color 0A
set PYTHONIOENCODING=utf-8
cls
cd /d "%~dp0"

REM This menu mixes TRAINING and SERVING, so it stays on the base env until the
REM full retrain runs under jackpot_gpu: TF 2.10 cannot open the current
REM models/*.keras (Keras 3 zip), and predict would silently lose every neural
REM member. Flip to `call "%~dp0activate_env.bat"` once the retrain is done.
REM (The old line here activated "gtrade_gpu", an env that does not exist, and
REM hid the failure with 2>nul - which is why the GPU sat unused for months.)
set PY_PATH=python
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
echo    [5R] Retrain chosen assets
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
echo    [AL] Autonomous cycle: search, A/B, adopt   [ALS] Its stage / stop it
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
if /i "%choice%"=="5R" goto train_assets
if /i "%choice%"=="AG" goto adopt_genome
if /i "%choice%"=="AS" goto adopt_show
if /i "%choice%"=="AR" goto adopt_revert
if /i "%choice%"=="RS" goto auto_research
if /i "%choice%"=="ALS" goto auto_loop_status
if /i "%choice%"=="AL" goto auto_loop
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
REM Mixed on purpose: data and the dashboard stay on the base env, only the
REM training step moves to the GPU one. Until the migration retrain has covered
REM every asset the dashboard will warn about champions it cannot read, which is
REM the correct signal rather than a silent downgrade.
python data_engine.py
cmd /c ""%~dp0run_in_env.bat" python train_hybrid.py"
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
REM Training runs in the GPU environment: it is many times faster there, and the
REM weights it writes are only readable by that environment anyway.
cmd /c ""%~dp0run_in_env.bat" python train_hybrid.py"
pause
goto menu

:train_assets
cls
echo Переобучение ВЫБРАННЫХ активов, в GPU-окружении, как и всякое обучение.
echo.
echo Зачем: реестр чемпионов пишется один раз в конце прогона, а файлы моделей
echo по ходу. Прерванный прогон поэтому оставляет активы, где файл новее записи
echo в реестре, и такой актив выпадает из сигналов с ошибкой о числе признаков.
echo Переобучение этих активов переписывает и файлы, и запись разом.
echo.
set /p TR_ASSETS="Активы через запятую (QCOM,MU,EURUSD), Enter = отмена: "
if "%TR_ASSETS%"=="" goto menu
echo.
echo [Retrain] %TR_ASSETS%
set "GTRADE_ASSETS=%TR_ASSETS%"
cmd /c ""%~dp0run_in_env.bat" python train_hybrid.py"
REM Снять переменную, иначе следующий пункт меню обучит только эти активы.
set "GTRADE_ASSETS="
pause
goto menu

:train_chunked
cls
echo Chunked trainer: one fresh process per chunk, resumable, and a champion
echo changes only if the new model beats it. Add --force-promote only to rebuild
echo a baseline or repair registry metadata.
echo.
echo Runs in the GPU environment. Serving still reads the OLD weights until this
echo finishes for every asset, so let it complete before switching predict over.
echo.
REM train_chunked spawns train_hybrid through sys.executable, so the whole chain
REM inherits the interpreter this line picks.
cmd /c ""%~dp0run_in_env.bat" python train_chunked.py"
pause
goto menu

:auto_research
cls
echo Hands over to auto_research.bat, which has its own menu (mode, proposer,
echo budget, objective, score basis). It never touches production: candidates
echo train into temp dirs and nothing is adopted without you.
echo.
REM  setlocal so the agent's GTRADE_* choices (extra columns,
REM  objective) die with it. Without this they would leak into whatever the
REM  main menu runs next, and a later Train would silently pick them up.
setlocal
call auto_research.bat
endlocal
goto menu

:auto_loop
cls
echo =======================================================
echo   AUTONOMOUS CYCLE  (search - A/B - adopt)
echo =======================================================
echo Runs the three phases in sequence and keeps cycling until something is
echo adopted, a phase fails, or you stop it. A failed A/B is not an ending: the
echo next cycle takes the next candidate, and when none is left it searches again.
echo.
echo It STOPS before the retrain and prints the full report plus the genome.
echo Production is untouched until you run the retrain yourself.
echo.
echo Stop it later with [ALS], or from any prompt: python auto_loop.py --stop
echo Its stage is also on the /research page while it runs.
echo.
REM  setlocal so the campaign's GTRADE_* choices die with the run. The main menu
REM  stays on the base env, and a leaked score basis or label mode would be
REM  picked up by a later Train from this same window.
setlocal
REM  The GPU env must be active BEFORE python starts: auto_loop spawns every
REM  train_hybrid through sys.executable, so the whole tree inherits this
REM  interpreter and this PATH, which is where the CUDA/cuDNN DLLs live. Without
REM  it the entire night runs on the CPU and nobody notices until morning.
call "%~dp0activate_env.bat"

echo [0] Campaign. The score basis and the objective decide HOW a result is
echo     judged, so they are frozen once a campaign starts - choosing them after
echo     seeing a verdict is a search for a verdict that passes, not a
echo     measurement. That is why they are asked here and nowhere else.
echo     1 = continue the current campaign (default)
echo     2 = start a NEW one (sets the search archive aside and re-freezes)
set "NEWC=1"
set /p "NEWC=    choice [1]: "
set "NEWCFLAG="
if not "%NEWC%"=="2" goto :al_director

echo.
echo [0a] Score basis: WHICH number the objective is applied to.
echo     1 = net_auc  (default) the nets' own AUC over all folds. The basis for
echo         any neural work: a rank statistic, so it does not inherit the Score's
echo         irreproducibility on this GPU (same seed, 0.45 to 1.52 apart).
echo     2 = ens_auc  ensemble AUC as a level. Use when a change moves BOTH
echo         learners, where net_gain would reward simply damaging CatBoost.
echo     3 = net_gain ensemble AUC minus CatBoost's: what the nets add. Use when
echo         the nets are given a target other than direction.
echo     4 = raw      the ensemble Score (a backtest of discrete signals).
echo     5 = neural   ensemble Score minus a CatBoost-only run.
set "BAS=1"
set /p "BAS=    choice [1]: "
set "GTRADE_AR_SCORE_BASIS=net_auc"
if "%BAS%"=="2" set "GTRADE_AR_SCORE_BASIS=ens_auc"
if "%BAS%"=="3" set "GTRADE_AR_SCORE_BASIS=net_gain"
if "%BAS%"=="4" set "GTRADE_AR_SCORE_BASIS=raw"
if "%BAS%"=="5" set "GTRADE_AR_SCORE_BASIS=neural"
REM  The screen and the illumination are DERIVED from the basis, not asked. On a
REM  net basis the CB-only screen stubs every net to a constant, so every
REM  candidate screens identically and net levers get thrown away on CatBoost's
REM  opinion. On a Score basis the reverse: full illumination would rank noise,
REM  because net training does not reproduce here. auto_loop refuses either
REM  combination anyway; deriving it means the menu cannot build one.
set "GTRADE_AR_SCREEN=0"
set "GTRADE_AR_ILLUM=full"
if "%BAS%"=="4" set "GTRADE_AR_SCREEN=1"
if "%BAS%"=="4" set "GTRADE_AR_ILLUM=cb"
if "%BAS%"=="5" set "GTRADE_AR_SCREEN=1"
if "%BAS%"=="5" set "GTRADE_AR_ILLUM=cb"

echo.
echo [0b] Objective: how the per-asset held-out lifts reduce to ONE number.
echo     1 = mean (default)   2 = min (lift the floor)   3 = median (robust)
echo     4 = cvar (mean of the worst quarter)   5 = sharpe (consistency)
echo     6 = trimmed (no extremes)
set "OBJ=1"
set /p "OBJ=    choice [1]: "
set "GTRADE_AR_OBJECTIVE=mean"
if "%OBJ%"=="2" set "GTRADE_AR_OBJECTIVE=min"
if "%OBJ%"=="3" set "GTRADE_AR_OBJECTIVE=median"
if "%OBJ%"=="4" set "GTRADE_AR_OBJECTIVE=cvar"
if "%OBJ%"=="5" set "GTRADE_AR_OBJECTIVE=sharpe"
if "%OBJ%"=="6" set "GTRADE_AR_OBJECTIVE=trimmed_mean"
set "NEWCFLAG=--new-campaign"
echo.
echo     New campaign: basis %GTRADE_AR_SCORE_BASIS%, objective %GTRADE_AR_OBJECTIVE%,
echo     screen %GTRADE_AR_SCREEN%, illumination %GTRADE_AR_ILLUM% (both derived from the basis).
echo     The search archive will be set aside as _qd_archive.json.bak.
:al_director

echo.
echo [1] Campaign director: an LLM picks the axis, label and budget each cycle.
echo     It can never touch the score basis or the objective - those are frozen
echo     for the campaign, and only a written new-campaign request can move them.
echo     1 = off (default: the campaign in auto_loop.py is used as written)
echo     2 = on
set "DIR=1"
set /p "DIR=    choice [1]: "
set "GTRADE_AR_DIRECTOR=0"
if "%DIR%"=="2" set "GTRADE_AR_DIRECTOR=1"

echo.
echo [2] Search proposer: what suggests each candidate genome INSIDE a search.
echo     Separate from the director, which chooses what the search is about.
echo     1 = evolutionary (default, no LLM, no API key)
echo     2 = LLM-guided (adds the "llm" arm to the RL scheduler)
set "PRP=1"
set /p "PRP=    choice [1]: "
set "GTRADE_AR_PROPOSER=evolutionary"
if "%PRP%"=="2" set "GTRADE_AR_PROPOSER=llm"

echo.
echo [3] Research wiki: after each run an LLM distils the findings journal into
echo     topic pages the proposer then reads, so learning compounds across runs.
echo     1 = on (default)   2 = off
set "WK=1"
set /p "WK=    choice [1]: "
set "GTRADE_AR_WIKI=1"
if "%WK%"=="2" set "GTRADE_AR_WIKI=0"

REM  THREE consumers can need a model: the director, the search proposer and the
REM  wiki. Ask once, and ask whenever ANY of them is on. The model used to be
REM  asked for under the director alone, so a default run - director off, wiki on
REM  - still called an LLM every cycle, with the provider and model coming from
REM  .env instead of from this menu.
set "NEEDLLM=0"
if "%GTRADE_AR_DIRECTOR%"=="1" set "NEEDLLM=1"
if "%GTRADE_AR_PROPOSER%"=="llm" set "NEEDLLM=1"
if "%GTRADE_AR_WIKI%"=="1" set "NEEDLLM=1"
REM  Set unconditionally and never left blank: cmd's  set "VAR="  DELETES the
REM  variable, and the agent's load_dotenv then refills a MISSING key from .env,
REM  which is how a 17 GB model once got pinned on a 15.7 GB machine.
set "GTRADE_AR_LLM=ollama"
set "GTRADE_AR_LLM_MODEL=auto"
set "GTRADE_AR_LLM_MAX_TOKENS=8000"
set "GTRADE_AR_LLM_TIMEOUT=3600"
if "%NEEDLLM%"=="0" goto :al_budget

echo.
echo [4] Which model serves them?
echo     1 = local Ollama (default, free, nothing leaves the machine)
echo     2 = Anthropic API (needs ANTHROPIC_API_KEY)
echo     3 = OpenAI API (needs OPENAI_API_KEY)
set "DLM=1"
set /p "DLM=    choice [1]: "
if "%DLM%"=="2" set "GTRADE_AR_LLM=anthropic"
if "%DLM%"=="3" set "GTRADE_AR_LLM=openai"
if "%DLM%"=="1" echo.
if "%DLM%"=="1" echo     Installed local models:
if "%DLM%"=="1" python -m core.llm_proposer --list-ollama
if "%DLM%"=="1" echo     Enter = auto-detect ^(first gemma, else first installed^).
set /p "GTRADE_AR_LLM_MODEL=    model name [auto]: "
echo.
echo     Seconds allowed for ONE call. A large local model on CPU needs far more
echo     than the 600s SDK default, and a timeout is not retried, so a value that
echo     is too small costs that model its whole contribution for the run.
set /p "GTRADE_AR_LLM_TIMEOUT=    timeout seconds [3600]: "
:al_budget

echo.
echo [5] Search iterations per cycle: how many NEW genomes each search phase
echo     trains and places in the archive. Nothing already in the tried registry
echo     is re-tested, so the budget only buys unseen candidates. At the campaign
echo     settings a genome costs roughly 9 minutes, so 15 is about 2 hours.
set "ALB=15"
set /p "ALB=    iterations [15]: "
echo.
echo [6] Deadline in hours. 0 = none, which is what "keep going until something
echo     is adopted" means. Any number stops it early, even mid-campaign.
set "ALH=0"
set /p "ALH=    hours [0]: "

echo.
echo ------------------------------------------------------------
if "%GTRADE_AR_DIRECTOR%"=="1" echo   director=on   proposer=%GTRADE_AR_PROPOSER%   wiki=%GTRADE_AR_WIKI%
if not "%GTRADE_AR_DIRECTOR%"=="1" echo   director=off   proposer=%GTRADE_AR_PROPOSER%   wiki=%GTRADE_AR_WIKI%
if "%NEEDLLM%"=="1" echo   llm=%GTRADE_AR_LLM%   model=%GTRADE_AR_LLM_MODEL%   timeout=%GTRADE_AR_LLM_TIMEOUT%s
if "%NEEDLLM%"=="0" echo   llm=not used by this configuration
if "%NEWC%"=="2" echo   NEW campaign: basis=%GTRADE_AR_SCORE_BASIS%  objective=%GTRADE_AR_OBJECTIVE%
if not "%NEWC%"=="2" echo   campaign=continuing the frozen one (see [ALS] for what it is)
echo   budget=%ALB% per cycle   deadline=%ALH% h
echo ------------------------------------------------------------
set "GO=Y"
set /p "GO=Start? [Y/n]: "
if /i "%GO%"=="n" goto :al_done
python auto_loop.py --budget %ALB% --hours %ALH% %NEWCFLAG%
:al_done
endlocal
pause
goto menu

:auto_loop_status
cls
echo Where the autonomous cycle stands. The same stage is on the /research page.
echo.
python auto_loop.py --status
echo.
echo   [1] back (default)   [2] ask a running loop to stop after its phase
set "ALS=1"
set /p "ALS=    choice [1]: "
if "%ALS%"=="2" python auto_loop.py --stop
pause
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
REM Same environment as the run below, so the picker and the measurement agree
REM about the feature space and the training cache.
cmd /c ""%~dp0run_in_env.bat" python ab_build.py"
pause
goto menu

:ab_run
cls
echo This trains the holdout once per arm. On the GPU environment that is hours,
echo not the day it used to take on CPU.
echo Do not start it while a retrain is running: they compete for RAM and cores.
echo Stop the scheduler and do not run data_engine while it works - new bars
echo mid-run make the arms measure different windows.
echo.
REM Runs in a CHILD cmd so the GPU environment does not leak into this menu:
REM serving stays on the base env until the full retrain. It matters for more
REM than speed - the training cache is keyed without the environment, so a run
REM started on base python could reuse rows trained on the GPU and compare two
REM arms measured under different TensorFlow builds.
cmd /c ""%~dp0run_in_env.bat" python ab_build.py --run"
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
REM Tuning is a training-class job: it writes models/optuna_params.json, no
REM weights, so the environment is a speed choice only (6 CatBoost threads).
cmd /c ""%~dp0run_in_env.bat" python optuna_tune.py"
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
