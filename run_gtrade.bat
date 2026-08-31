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
echo.
echo         /\                 _____ _____  _____ _____
echo        /  \   .-'''-.     ^|  _  ^|_   _^|^|  _  ^|  _  ^|
echo       /    \ /       \    ^| ^|_^| ^| ^| ^|  ^| ^|_^| ^| ^|_^| ^|
echo      /  /\  \  .-. .-.^\   ^|  _  ^| ^| ^|  ^|  _  ^|  _  ^|
echo     /  /  \  \(o o^) ^|  ^\  ^|_^| ^|_^| ^|_^|  ^|_^| ^|_^|_^| ^|_^|
echo    /__/    \__\ ^^^   /__/       A T R A T U S
echo         \      /  ^\_/
echo          \____/          signals, levels, and the evidence for both
echo.
echo =======================================================
echo.
echo  DAILY
echo    [1] Full Cycle      [3] Predict (Radar)     [4] Data Update
echo    [2] Dashboard       [WU] Web UI (FastAPI)
echo.
echo  TRAINING
echo    [5] Train Models    [5C] Chunked            [5R] Chosen assets
echo    [5F] Fill in / repair champions             [T] Optuna Tune
echo.
echo  SIGNALS
echo    [6] Backtest        [M] Model Health        [E] Export CSV
echo    [L] Signal Log      [H] HTML Report         [Q] Equity Curve
echo    [SG] Publish live signals to the site
echo.
echo  ANALYTICS
echo    [N] News Analyzer   [D] News Digest         [R] Regime Detector
echo    [C] Correlation     [WL] Watchlist          [P] Paper Trading
echo    [W1] Top-5 90d      [W2] Top-10 90d         [W3] Top-5 180d
echo    [W4] Top-5 Kelly    [W5] Custom assets
echo.
echo  RESEARCH
echo    [RS] Auto-research agent (own menu)         [AN] Analyst agent
echo    [AL] Autonomous cycle: search, A/B, adopt   [ALS] Its stage / stop it
echo    [LC] Daily loop cycle
echo.
echo  POLICIES
echo    [TP] Timing rules   [TB] Timing: fitted-Q challenger
echo    [TL] Trade levels   [TO] Timing: one online tick
echo    [SZ] Position sizing                        [DR] Direction rule
echo    [RC] Recalibrate live probabilities         [OS] Refit on unscored
echo    [PS] How the policies did on LIVE signals   [TR] Timing replay
echo.
echo  GENOME
echo    [AG] Adopt          [AS] What is adopted    [AR] Revert
echo    [ABC] Configure an A/B                      [ABR] Run it
echo.
echo  SERVICES
echo    [7] Telegram Bot    [8] Scheduler           [9] DB Audit
echo    [F] DB Fix          [B] DB Backup           [I] Install/Repair
echo.
echo    [0] EXIT
echo.
echo =======================================================
set /p choice="Select: "

if "%choice%"=="1" goto full_run
if "%choice%"=="2" goto dashboard
if /i "%choice%"=="WU" goto webui_root
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
if /i "%choice%"=="5F" goto fill_champions
if /i "%choice%"=="AG" goto adopt_genome
if /i "%choice%"=="AS" goto adopt_show
if /i "%choice%"=="AR" goto adopt_revert
if /i "%choice%"=="RS" goto auto_research
if /i "%choice%"=="AN" goto analyst
if /i "%choice%"=="ALS" goto auto_loop_status
if /i "%choice%"=="AL" goto auto_loop
if /i "%choice%"=="LC" goto loop_cycle
if /i "%choice%"=="RC" goto recalibrate
if /i "%choice%"=="TP" goto timing_policy
if /i "%choice%"=="TL" goto levels_policy
if /i "%choice%"=="PS" goto policy_status
if /i "%choice%"=="TB" goto timing_stage_b
if /i "%choice%"=="TO" goto timing_online
if /i "%choice%"=="SZ" goto sizing_policy
if /i "%choice%"=="DR" goto direction_policy
if /i "%choice%"=="OS" goto out_of_sample
if /i "%choice%"=="TR" goto timing_replay
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

:webui_root
set "WU_PATH="
goto webui

:analyst_watch
set "WU_PATH=/analyst"
goto webui

:webui
cls
REM  WU_PATH is the page to land on, empty for the dashboard. Every entry
REM  point sets it before jumping here, INCLUDING the one that wants the
REM  root: the menu loops, so a value left over from an earlier visit would
REM  otherwise send the next [WU] to whatever page was opened last.
echo [Web UI] FastAPI on http://127.0.0.1:8000%WU_PATH%  (Ctrl+C to stop)
REM  The browser used to open on the line BEFORE uvicorn started, so the first
REM  paint was always "connection refused" and the UI read as broken. Hand the
REM  open to a detached shell that waits for the server to come up.
start "" /b cmd /c "timeout /t 4 /nobreak >nul&start http://127.0.0.1:8000%WU_PATH%"
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

:fill_champions
cls
echo Two different illnesses, two different cures. Pick one.
echo.
echo [1] Assets with NO champion. Nothing was ever trained for them, so every
echo     policy fit rebuilds its environment from champion probabilities,
echo     finds nothing, and skips them WITHOUT saying which names are gone.
echo     A plain run: there is no incumbent, so the first model promotes itself.
echo.
echo [2] Assets whose neural champion does not load here. They still serve, on
echo     fewer members than the registry claims, and no error says so. Repair
echo     NEEDS force-promote: without it the champion is only rewritten when the
echo     challenger wins, and on a loss the broken file survives untouched.
echo.
echo Both run through the chunked trainer: one fresh process per chunk, so RAM
echo stays flat over a long run and an interrupted pass resumes. The asset list
echo is worked out for you - nothing to paste.
echo.
echo [3] Both in ONE pass. force-promote is on, which the degraded half needs
echo     and which decides nothing for the others: train_hybrid promotes when
echo     there is no registry entry to compare against, so an asset that never
echo     had a champion is promoted with the flag or without it.
echo.
echo The list is worked out HERE, in base, and only the training is handed to
echo the GPU environment. Which champions are broken is a question about the
echo loader serving uses; asked inside the training env every Keras 3 champion
echo answers "broken" and a force-promote pass would rebuild the whole project.
echo.
set /p FC_WHICH="[1] fill in, [2] repair degraded, [3] both, Enter = cancel: "
if "%FC_WHICH%"=="" goto menu
echo.
if "%FC_WHICH%"=="1" set "FC_KIND=missing"
if "%FC_WHICH%"=="2" set "FC_KIND=degraded"
if "%FC_WHICH%"=="3" set "FC_KIND=all"
if "%FC_KIND%"=="" goto menu
echo Working out the %FC_KIND% list (opens every champion, takes a minute)...
python model_health.py --list %FC_KIND% --out _repair_list.txt
if errorlevel 1 (
  echo.
  echo The list could not be built, so nothing was trained.
  set "FC_KIND="
  pause
  goto menu
)
echo.
if "%FC_WHICH%"=="1" cmd /c ""%~dp0run_in_env.bat" python train_chunked.py --assets-file _repair_list.txt"
if not "%FC_WHICH%"=="1" cmd /c ""%~dp0run_in_env.bat" python train_chunked.py --assets-file _repair_list.txt --force-promote"
set "FC_KIND="
set "FC_WHICH="
echo.
echo Serving reads the new weights only after this finishes. Re-check with [M],
echo then refit the policies on the new assets with [OS].
pause
goto menu

:timing_replay
cls
echo Walks the baseline, the adopted Stage-A rules and the adopted Stage-B Q
echo over the held-out slice and asks of every decision whether it turned out
echo right. No money, no positions taken - the same hit-or-miss reading the
echo signal itself gets, applied to what a timing layer actually decides.
echo.
echo In a position, a hit is the bar going the way the position faces. Flat
echo while the raw signal wanted in, a hit is the skipped trade not paying.
echo Bars where nothing was chosen are not counted either way.
echo.
echo The last two columns are trades and net after cost, from the SAME
echo evaluator the ADOPT verdict was read from. Accuracy and net can point
echo different ways: holding longer pays for fewer legs.
echo.
echo Nothing is fitted and nothing is written. Slow - it rebuilds every
echo asset's scorable history first.
echo.
set /p RP_ASSETS="Assets (list or ALL), Enter = cancel: "
if "%RP_ASSETS%"=="" goto menu
set "RP_SEL=--assets %RP_ASSETS%"
if /i "%RP_ASSETS%"=="all" set "RP_SEL="
echo.
python train_timing.py --replay %RP_SEL%
set "RP_ASSETS="
set "RP_SEL="
pause
goto menu

:out_of_sample
cls
echo An adopted rule tested again on the data that adopted it proves nothing.
echo These are the assets each policy's gate has NEVER scored, so a refit
echo restricted to them is an independent replication, not a second reading.
echo.
echo Read the fitted parameters against the adopted ones. The same shape
echo appearing on assets the first fit never saw is the evidence; a verdict on
echo its own is not, because this refits rather than transferring the rule.
echo.
python policy_status.py --unseen
echo.
echo   [1] timing Stage A   [2] sizing   [3] trade levels
echo   [4] timing Stage B, ONE pre-registered horizon (--iters 1, no
echo       multiplicity to correct - the only form in which Stage B reopens)
echo.
set /p OS_WHICH="Which, Enter = cancel: "
if "%OS_WHICH%"=="" goto menu
echo Assets: paste a list, or type ALL for every asset in the map.
echo.
echo   ALL is a REFIT, not a replication. It re-reads the data the current
echo   rule was fitted on, so its verdict is not out-of-sample evidence for
echo   anything already adopted - it is the new production fit. Use it when
echo   you mean "refit on everything there now is", and a restricted list when
echo   you mean "check the adopted rule against assets it never saw".
echo.
set /p OS_ASSETS="Assets (list or ALL), Enter = cancel: "
if "%OS_ASSETS%"=="" goto menu
set "OS_SEL=--assets %OS_ASSETS%"
if /i "%OS_ASSETS%"=="all" set "OS_SEL="
if /i "%OS_ASSETS%"=="all" echo [OS] every asset in the map; the fitters skip any without a champion.
set /p OS_BUDGET="Search iterations (Enter = 300): "
if "%OS_BUDGET%"=="" set OS_BUDGET=300
echo.
if "%OS_WHICH%"=="1" python train_timing.py %OS_SEL% --budget %OS_BUDGET%
if "%OS_WHICH%"=="2" python train_sizing.py %OS_SEL% --budget %OS_BUDGET%
if "%OS_WHICH%"=="3" python train_levels.py %OS_SEL% --budget %OS_BUDGET%
if "%OS_WHICH%"=="4" python train_timing.py --stage b --iters 1 %OS_SEL%
set "OS_WHICH="
set "OS_ASSETS="
set "OS_SEL="
set "OS_BUDGET="
echo.
echo A gate line reading "N asset(s) dropped as unscorable" is not noise: those
echo assets had an arm with too few trades to judge and were left out of the
echo effect size on purpose.
pause
goto menu

:train_assets
cls
echo Retrain CHOSEN assets, in the GPU environment like all training.
echo.
echo Why: an interrupted run leaves assets whose model files are newer than
echo their champion-registry entry, because the files land per asset while the
echo registry used to land once at the end. Such an asset drops out of the
echo signals with a feature-count error; retraining rewrites both together.
echo.
echo Assets in that state right now:
python model_health.py --mismatched
echo.
set /p TR_ASSETS="Assets, comma separated, Enter = cancel: "
if "%TR_ASSETS%"=="" goto menu
echo.
REM Repairing a stale entry NEEDS force-promote. Without it a champion is only
REM rewritten when the challenger wins, and on a loss the orphaned file and the
REM entry that disagrees with it both survive untouched - the asset stays broken.
set /p TR_FORCE="Force promote? Required to repair the list above. y/N: "
set "GTRADE_ASSETS=%TR_ASSETS%"
if /i "%TR_FORCE%"=="y" set "GTRADE_FORCE_PROMOTE=1"
echo.
echo [Retrain] %TR_ASSETS%   force-promote: %TR_FORCE%
cmd /c ""%~dp0run_in_env.bat" python train_hybrid.py"
REM Clear both, or the next menu item would train only these assets, and would
REM replace every champion it touched regardless of score.
set "GTRADE_ASSETS="
set "GTRADE_FORCE_PROMOTE="
set "TR_FORCE="
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

:analyst
cls
echo  ANALYST AGENT - an opinion formed without seeing the ensemble's.
echo.
echo    [S] Score      standings against the three baselines, and the verdict
echo    [B] Backfill   fill outcomes whose horizon has elapsed
echo    [F] Fit table  refit payoff_stats.json from prediction_log
echo    [R] Run        one judgment per eligible asset  (COSTS MONEY: one LLM
echo                   call per asset, plus an earnings scan over the map)
echo    [W] Watch      start the Web UI on the analyst page
echo.
set "an_choice="
set /p an_choice="Choose, Enter = back: "
if /i "%an_choice%"=="S" goto analyst_score
if /i "%an_choice%"=="B" goto analyst_backfill
if /i "%an_choice%"=="R" goto analyst_run
if /i "%an_choice%"=="F" goto analyst_fit
if /i "%an_choice%"=="W" goto analyst_watch
goto menu

:analyst_score
python analyst.py score
pause
goto analyst

:analyst_backfill
python analyst.py backfill
pause
goto analyst

:analyst_run
echo.
echo This spends money: one LLM call per asset judged. GTRADE_ANALYST=0
echo disables the agent entirely if you would rather it never ran.
echo.
set "an_assets="
set /p an_assets="Assets (comma-separated), Enter = watchlist + earnings today: "
echo.
echo    [1] anthropic   [2] openai   [3] ollama   Enter = whatever .env says
set "an_llm="
set /p an_llm="Model provider: "
set "an_flag="
if "%an_llm%"=="1" set "an_flag=--llm anthropic"
if "%an_llm%"=="2" set "an_flag=--llm openai"
if "%an_llm%"=="3" set "an_flag=--llm ollama"
set "an_name="
if not "%an_flag%"=="" set /p an_name="Model name, Enter = provider default: "
if not "%an_name%"=="" set "an_flag=%an_flag% --model %an_name%"
echo.
set "an_ok="
set /p an_ok="Type YES to run: "
if /i not "%an_ok%"=="YES" goto analyst
if "%an_assets%"=="" (python analyst.py run %an_flag%) else (python analyst.py run --assets "%an_assets%" %an_flag%)
pause
goto analyst

:analyst_fit
echo.
echo train_payoff.py OVERWRITES payoff_stats.json on every run. If this fit
echo covers fewer assets than the last one, the fuller table is gone.
echo.
set "fit_ok="
set /p fit_ok="Type YES to refit: "
if /i not "%fit_ok%"=="YES" goto analyst
python train_payoff.py
pause
goto analyst

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
echo     2 = start a NEW one (re-freezes; the archive is kept unless the
echo         SEARCH basis itself moves)
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
REM  The SCREEN stays derived: on a net basis the CB-only screen stubs every net
REM  to a constant, so every candidate screens identically and net levers get
REM  thrown away on CatBoost's opinion. The ILLUMINATION is asked below, but only
REM  where both answers are real: on a Score basis full illumination would rank
REM  noise, because net training does not reproduce here, and auto_loop refuses
REM  that pairing - so 4 and 5 keep cb and skip the question.
set "GTRADE_AR_SCREEN=0"
set "GTRADE_AR_ILLUM=full"
if "%BAS%"=="4" set "GTRADE_AR_SCREEN=1"
if "%BAS%"=="4" set "GTRADE_AR_ILLUM=cb"
if "%BAS%"=="5" set "GTRADE_AR_SCREEN=1"
if "%BAS%"=="5" set "GTRADE_AR_ILLUM=cb"
if "%BAS%"=="4" goto :illum_done
if "%BAS%"=="5" goto :illum_done

echo.
echo [0a2] Search illumination: what the ARCHIVE is scored on, i.e. which
echo     genomes ever become elites.
echo     1 = full (default) trains the tier assets with REAL nets, about 545s a
echo         genome. The only setting under which a net lever can become an
echo         elite: the basis you just picked decides what gets illuminated.
echo     2 = cb   the CatBoost-only screen, about 43s a genome, 12x cheaper.
echo         Every net member is stubbed to a constant 0.5, so every elite is a
echo         pure CatBoost pick and the basis only re-scores the final gate.
echo     Clear _qd_archive.json when you switch: Score-scale fitness never
echo     loses to AUC-scale fitness, so the two do not share an archive.
set "ILL=1"
set /p "ILL=    choice [1]: "
if "%ILL%"=="2" set "GTRADE_AR_ILLUM=cb"
:illum_done

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
echo [0c] Decision basis: WHICH number an ADOPTION is judged on. It need not be
echo     the one the search optimises. The search basis is picked for signal to
echo     noise; this one decides what counts as an improvement to production.
echo     Measured 2026-08-18: over one holdout, mean Net_AUC +0.036 while mean
echo     Score was -1.85, rank correlation -0.24. The A/B passed and the retrain
echo     it authorised kept the champion on 23 of the first 29 assets.
echo     1 = raw Score, what train_hybrid promotes champions on (recommended)
echo     2 = same as the search basis (the behaviour before this existed)
set "DEC=1"
set /p "DEC=    choice [1]: "
set "GTRADE_AR_DECISION_BASIS=raw"
if "%DEC%"=="2" set "GTRADE_AR_DECISION_BASIS="
REM  Not asked, because there is one right answer. tier_neural_floor() is
REM  2 * neural_floor(), and neural_floor() is -inf on every net basis, so the
REM  tier check that refuses a genome for starving the nets never fires on the
REM  campaigns that most need it. Writing the documented default out re-arms
REM  it. It filters what reaches the gate and never changes what a result must
REM  clear there.
set "GTRADE_AR_TIER_NEURAL_MIN=-1.0"

echo.
echo [0d] Gate size: how many assets a verdict is measured over. dScore has a
echo     standard deviation near 3.74 per asset, so a gate of n resolves about
echo     2.8 * 3.74 / sqrt(n):
echo         14 -^> +2.80      40 -^> +1.66      80 -^> +1.17
echo     The one genome ever adopted, A, measured +1.63 - under the resolution
echo     of the 14 it was measured with. Both gates move together: the search's
echo     own (GTRADE_AR_HELDOUT, which decides what is even offered) and the
echo     final A/B's. The search gate GROWS, keeping every asset already in it,
echo     so earlier runs stay comparable.
echo     Cost per arm, measured 2026-08-13: 33 min at 14, so ~95 at 40.
echo     1 = 40 (recommended)   2 = 14 (as before)   3 = 80   4 = other
echo     5 = neural: the 14 assets whose stacker actually leans on the nets.
echo         Measured from the ^|coef^| shares in models/*_meta.pkl: those
echo         assets give the neural members .54-.69 of the stacker's weight,
echo         against a .17 median on the standard gate, where ADA is .05 and
echo         MSFT .08. A neural change measured on the standard gate is
echo         measured mostly where the nets barely matter. Use this to SEE a
echo         neural effect, not to adopt one: the set is biased by
echo         construction, so a winner here still has to clear a normal gate.
set "GS=1"
set /p "GS=    choice [1]: "
if "%GS%"=="5" goto gate_neural
set "GATE_N=40"
if "%GS%"=="2" set "GATE_N=14"
if "%GS%"=="3" set "GATE_N=80"
if "%GS%"=="4" set /p "GATE_N=    assets [40]: "
if "%GATE_N%"=="" set "GATE_N=40"
set "GTRADE_AB_HOLDOUT_N=%GATE_N%"
echo     building the search gate list for %GATE_N% assets...
python ab_build.py --search-gate %GATE_N% --out _search_gate.txt
if errorlevel 1 (
  echo     could not build it; leaving the search gate as it was.
) else (
  set /p GTRADE_AR_HELDOUT=<_search_gate.txt
)
goto gate_done

:gate_neural
REM The neural-diagnostic holdout lives in auto_research.heldout_assets()
REM and was reachable only by setting the variable by hand, because the
REM block above overwrote it from _search_gate.txt on every campaign start.
REM Both gates stay on the same 14 assets so the search and the A/B read
REM the same set.
set "GTRADE_AR_HELDOUT=neural"
set "GTRADE_AB_HOLDOUT_N=14"
echo     search gate: neural-diagnostic set, 14 assets, for MEASURING nets.

:gate_done
echo.
echo     New campaign: search basis %GTRADE_AR_SCORE_BASIS%, objective %GTRADE_AR_OBJECTIVE%,
echo     screen %GTRADE_AR_SCREEN% (derived from the basis), illumination %GTRADE_AR_ILLUM%.
echo     The search archive is kept: only a move of the SEARCH basis sets it aside.
:al_director

echo.
echo [1] Campaign director: who picks the axis, mode, budget and the rest each
echo     cycle. None of them can touch the score basis or the objective - those
echo     are frozen for the campaign, and only a written new-campaign request
echo     can move them.
echo     1 = off (default: the campaign in auto_loop.py is used as written)
echo     2 = LLM
echo     3 = RL bandit over cycle recipes
echo     4 = alternate: LLM on even cycles, RL on odd (the comparison run)
set "DIR=1"
set /p "DIR=    choice [1]: "
set "GTRADE_AR_DIRECTOR=0"
set "GTRADE_AR_DIRECTOR_MODE=llm"
REM Parenthesised: in cmd an unparenthesised "if cond set A & set B" runs the
REM SECOND set unconditionally, which would put every run into RL mode.
if "%DIR%"=="2" set "GTRADE_AR_DIRECTOR=1"
if "%DIR%"=="3" ( set "GTRADE_AR_DIRECTOR=1" & set "GTRADE_AR_DIRECTOR_MODE=rl" )
if "%DIR%"=="4" ( set "GTRADE_AR_DIRECTOR=1" & set "GTRADE_AR_DIRECTOR_MODE=alternate" )

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
REM The RL director asks no model anything, so it does not by itself need one.
if "%GTRADE_AR_DIRECTOR_MODE%"=="llm" if "%GTRADE_AR_DIRECTOR%"=="1" set "NEEDLLM=1"
if "%GTRADE_AR_DIRECTOR_MODE%"=="alternate" set "NEEDLLM=1"
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
if "%GTRADE_AR_DIRECTOR%"=="1" echo   director=%GTRADE_AR_DIRECTOR_MODE%   proposer=%GTRADE_AR_PROPOSER%   wiki=%GTRADE_AR_WIKI%
if not "%GTRADE_AR_DIRECTOR%"=="1" echo   director=off   proposer=%GTRADE_AR_PROPOSER%   wiki=%GTRADE_AR_WIKI%
if "%NEEDLLM%"=="1" echo   llm=%GTRADE_AR_LLM%   model=%GTRADE_AR_LLM_MODEL%   timeout=%GTRADE_AR_LLM_TIMEOUT%s
if "%NEEDLLM%"=="0" echo   llm=not used by this configuration
if "%NEWC%"=="2" echo   NEW campaign: search=%GTRADE_AR_SCORE_BASIS%  decision=%GTRADE_AR_DECISION_BASIS%  objective=%GTRADE_AR_OBJECTIVE%
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

:policy_status
cls
echo What every policy layer concluded, and what it was worth on the signals
echo production actually sent. Two different claims, kept apart on purpose: a
echo backtest verdict is not a live reading.
echo.
echo It also writes policy_status.json, which is what the /research page shows,
echo so the page never has to read every asset's bars on a request.
echo.
set /p PS_DAYS="Reconcile the last N trading days (Enter = all): "
if "%PS_DAYS%"=="" (python policy_status.py) else (python policy_status.py --days %PS_DAYS%)
set "PS_DAYS="
pause
goto menu

:timing_stage_b
cls
echo Fits a Q challenger to the ADOPTED timing rules and gates it against them,
echo not against the baseline: the rules already beat the baseline, so beating
echo it again would prove nothing.
echo.
echo Prints the correlation between what the fit maximises and what the gate
echo reads, beside the verdict. A verdict that arrives with a flat or negative
echo correlation is a coincidence until it replicates.
echo.
set /p TB_ITERS="Q iterations, one horizon rung each (Enter = 6): "
if "%TB_ITERS%"=="" set TB_ITERS=6
cmd /c ""%~dp0run_in_env.bat" python train_timing.py --stage b --iters %TB_ITERS%"
set "TB_ITERS="
pause
goto menu

:timing_online
cls
echo One online tick: refit the timing Q on the newest data and accept it only
echo if it stays near the adopted rules and beats the generation in shadow.
echo.
echo The anchor is the rules, permanently, never the previous generation. Two
echo losing generations in a row halt the schedule back to the rules.
echo.
echo Self-collection: what fraction of assets have their transitions
echo generated by the CURRENT accepted Q instead of by the rules. At 0 the
echo Q only ever sees the rules trajectory plus 10%% flipped actions, so it
echo learns to correct THEIR mistakes and never the states its own policy
echo reaches. Whatever you pick, agreement is still measured against the
echo rules: only the data moves, never the trust region.
echo     The floor is 0.80 and the live generation sits at 0.811, so there
echo     is 0.011 of headroom. Raise this one step at a time and watch the
echo     agreement in the journal. A run that falls through the floor is
echo     REJECTED, which costs the tick but does not count as a rollback.
echo     1 = 0.00 off (default)   2 = 0.10   3 = 0.25   4 = other
set "TO_SHARE=1"
set /p "TO_SHARE=    choice [1]: "
set "TO_VAL=0.0"
if "%TO_SHARE%"=="2" set "TO_VAL=0.10"
if "%TO_SHARE%"=="3" set "TO_VAL=0.25"
if "%TO_SHARE%"=="4" set /p "TO_VAL=    fraction [0.10]: "
if "%TO_VAL%"=="" set "TO_VAL=0.10"
echo.
cmd /c ""%~dp0run_in_env.bat" python train_timing_online.py --self-share %TO_VAL%"
set "TO_SHARE="
set "TO_VAL="
pause
goto menu

:direction_policy
cls
echo Asks the widest question in the system: should the ensemble's direction be
echo followed at all, above a given confidence. Fitted on the earlier live days
echo and judged on the later ones, paired over assets. Following is in the
echo search space, so the incumbent can win.
echo.
echo Fitted on LIVE outcomes, not on reconstructed history, because the two
echo disagree about the sign of the relationship this rule conditions on.
echo.
echo Nothing is served. It writes direction_report.json and no serve path
echo reads it.
echo.
set /p DR_DAYS="Live days to use (Enter = 120): "
if "%DR_DAYS%"=="" set DR_DAYS=120
python train_direction.py --days %DR_DAYS%
set "DR_DAYS="
pause
goto menu

:sizing_policy
cls
echo Fits how big a position should be, given the side something else already
echo chose. Exposure is MATCHED before scoring, so a rule cannot win by simply
echo holding more: a constant size scores exactly like the unit position.
echo.
echo Nothing here reaches serving. Production sizing is Kelly plus the Taleb
echo cap and this does not touch it.
echo.
set /p SZ_BUDGET="Search iterations (Enter = 300): "
if "%SZ_BUDGET%"=="" set SZ_BUDGET=300
python train_sizing.py --budget %SZ_BUDGET%
set "SZ_BUDGET="
pause
goto menu

:levels_policy
cls
echo Fits the levels multipliers (entry zone and stop, in ATR) over the history
echo of every asset at once, and writes levels_policy.json ONLY if a held-out
echo slice agrees. Otherwise production keeps the levels it has.
echo.
echo The timing policy is frozen while this runs: it already passed its own
echo gate, and fitting both at once would hide which half earned the result.
echo.
set /p TL_BUDGET="Search iterations (Enter = 300): "
if "%TL_BUDGET%"=="" set TL_BUDGET=300
python train_levels.py --budget %TL_BUDGET%
set "TL_BUDGET="
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
echo.
echo === HOW WELL THE MODELS ACTUALLY DO ===
echo The live log is the honest answer and needs no correction: every row was
echo written before the bar it is scored against. Offline accuracy is not -
echo the champion is re-scored over the history it was fitted across.
echo.
python model_health.py --generations
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
echo [What-If] Top-5 signals, 90 days, equal weights...
python whatif_simulator.py --top 5 --days 90 --strategy equal
pause
goto menu

:whatif_top10
cls
echo [What-If] Top-10 signals, 90 days, equal weights...
python whatif_simulator.py --top 10 --days 90 --strategy equal
pause
goto menu

:whatif_180
cls
echo [What-If] Top-5 signals, 180 days, equal weights...
python whatif_simulator.py --top 5 --days 180 --strategy equal
pause
goto menu

:whatif_kelly
cls
echo [What-If] Top-5 signals, 90 days, Kelly weights...
python whatif_simulator.py --top 5 --days 90 --strategy kelly
pause
goto menu

:whatif_custom
cls
set /p WI_ASSETS="Assets, space separated (BTC ETH NVDA ...): "
set /p WI_DAYS="Days (Enter = 90): "
if "%WI_DAYS%"=="" set WI_DAYS=90
set /p WI_CAP="Capital USD (Enter = 10000): "
if "%WI_CAP%"=="" set WI_CAP=10000
echo.
echo [What-If] Assets: %WI_ASSETS% ^| Days: %WI_DAYS% ^| Capital: $%WI_CAP%
python whatif_simulator.py %WI_ASSETS% --days %WI_DAYS% --capital %WI_CAP%
pause
goto menu
