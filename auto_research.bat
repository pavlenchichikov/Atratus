@echo off
REM ===========================================================================
REM  AUTO-RESEARCH AGENT launcher (interactive menu).
REM  Answer the prompts (Enter = default) and the agent starts. It NEVER
REM  touches production: variants train into isolated temp dirs and nothing is
REM  auto-adopted - the agent only flags winners for a human. See README.
REM
REM  Cross-run memory: _ar_tried.json (never re-tests a candidate),
REM  _ar_eval_cache.json (base runs reused while the data is unchanged),
REM  _ar_findings.json (cumulative findings journal). Budget = NEW iterations
REM  per run, so periodic launches keep exploring fresh candidates.
REM
REM  Research wiki (menu item 5): a compounding, self-maintained knowledge base
REM  (GTRADE_AR_WIKI). After each run an LLM distills the findings journal into
REM  _ar_wiki/*.md topic pages the proposer then reads, so learning accumulates
REM  across runs instead of a sliding window. It uses the LLM backend, so pick an
REM  LLM proposer (or it defaults to Anthropic and needs ANTHROPIC_API_KEY). Off
REM  by default, so it is byte-identical. When on, this script also offers a wiki lint
REM  (reconcile contradictions + prune stale claims) after the run.
REM
REM  RL scheduler (menu item 7): a learned budget allocator over the QD child
REM  sources (GTRADE_AR_RL). A discounted Thompson bandit learns which of the
REM  nine emitters (feature/hyper/nets/tuning mutations, crossover, LLM,
REM  surrogate, CMA over the numeric genes, novelty targeting of empty niches)
REM  actually produce archive improvements, plus curiosity-based parent
REM  selection. The statistical adoption gate is untouched - the scheduler only
REM  decides what to TRY. Off by default (byte-identical uniform search); it
REM  self-disables within a run if it underperforms the uniform baseline.
REM  State: rl_scheduler_v1.json (posteriors are printed at run start/end).
REM
REM  Advanced knobs (screen, prune floor, QD sizes, seed, base URL, exhaustion
REM  cutoff) live below as set lines; the menu only asks the everyday questions.
REM ===========================================================================

cd /d "%~dp0"

REM The GPU env must be active BEFORE python starts: auto_research spawns every
REM train_hybrid child through sys.executable, so the whole tree inherits this
REM interpreter and this PATH (which is where the CUDA/cuDNN DLLs live).
call "%~dp0activate_env.bat"

REM == Advanced knobs (edit here; the menu does not ask about these) ===========
REM    The screen replaces the nets with a constant 0.5, so on a NET basis
REM    (GTRADE_AR_SCORE_BASIS=net_auc) every candidate screens identically and
REM    the screen carries no information - it only throws net levers away on
REM    CatBoost's opinion. Set to 0 for any net-basis run. 1 for Score runs.
set "GTRADE_AR_SCREEN=0"
REM    QD search illumination. "cb" = the historical cheap CatBoost-only screen
REM    (nets stubbed to 0.5, ~43s/genome) - fast, but every elite is then a pure
REM    CatBoost pick and no net lever can ever be found. "full" trains the tier
REM    assets with real nets (~545s/genome, 12x) so the net basis actually decides
REM    what gets illuminated. Use "full" ONLY with GTRADE_AR_SCORE_BASIS=net_auc:
REM    on the raw Score basis the nets do not reproduce on this GPU and the
REM    archive would rank noise. Clear _qd_archive.json when switching basis -
REM    Score-scale fitness (1.5-8.9) never loses to AUC-scale fitness (~0.01).
set "GTRADE_AR_ILLUM=full"
set "GTRADE_AR_SCREEN_MIN=0.0"
set "GTRADE_AR_PRUNE_MIN=8"
set "GTRADE_AR_QD_INIT=8"
set "GTRADE_AR_QD_FINAL=3"
REM    GTRADE_AR_SEED seeds the SEARCH (which genomes get proposed).
REM    GTRADE_SEED seeds the TRAINING (weights, shuffling, CatBoost). Two
REM    different things despite the names. Change GTRADE_SEED to re-roll a whole
REM    run: that is how you measure how much of a result is seed noise, and it
REM    gives the re-roll its own eval-cache namespace so it cannot read the
REM    previous roll's rows.
set "GTRADE_AR_SEED=42"
set "GTRADE_SEED=1000"
REM    Floor on neural_lift for adoption (Score scale). A candidate that clears
REM    the Score bar but sinks the nets below this is rejected instead of merely
REM    reported. Blank = default -0.5.
set "GTRADE_AR_NEURAL_FLOOR="
REM    Adoption floor on the net_auc basis (AUC units, not Score). Blank = 0.005.
set "GTRADE_AR_ADOPT_AUC="
set "AR_PRESCREEN_MIN=0.02"
set "GTRADE_AR_QD_LLM_P=0.3"
set "GTRADE_AR_QD_MAX_MISSES=5"
REM    Base URL override (blank = provider default / Ollama localhost):
set "GTRADE_AR_LLM_BASE_URL="
REM    Model override ("auto" = auto-detect for Ollama / provider default);
REM    the menu sets this for you when you pick a local or OpenAI model.
REM    NOT blank on purpose: cmd's  set "VAR="  DELETES the variable, so
REM    load_dotenv refills it from .env - which silently overrode this menu
REM    and pinned a 17 GB model on a 15.7 GB machine (2026-08-14).
set "GTRADE_AR_LLM_MODEL=auto"

echo ============================================================
echo   AUTO-RESEARCH  (Enter = default)
echo ============================================================
echo.
echo [0] Action:  1 = search for new candidates (default)
echo              2 = re-gate stored candidates under the current gate (reuse past runs)
set "ACT=1"
set /p "ACT=    choice [1]: "
if "%ACT%"=="2" goto :regate

echo.
echo [1] Mode (type the number, or an axes name/list directly):
echo     1 = qd (MAP-Elites quality-diversity, the flagship; genome now also
echo         carries hyperparameter, net-hygiene and triple-barrier genes)
echo     2 = features (DSL forward-selection)
echo     3 = labeling,pruning (rel_median windows + triple_barrier horizons; drops)
echo         (the weighting axis is deliberately NOT here: axes do not compose, and
echo         its candidates are no-ops unless GTRADE_LABEL_MODE is a multi-bar label,
echo         so run it via option 5 with that env already set)
echo     4 = hyper,nets,thresholds,regime (model + tuning levers)
echo     5 = custom (type your own axes list)
set "MODE=1"
set /p "MODE=    choice [1]: "
set "GTRADE_AR_AXES="
if "%MODE%"=="1" set "GTRADE_AR_AXES=qd"
if "%MODE%"=="2" set "GTRADE_AR_AXES=features"
if "%MODE%"=="3" set "GTRADE_AR_AXES=labeling,pruning"
if "%MODE%"=="4" set "GTRADE_AR_AXES=hyper,nets,thresholds,regime"
if "%MODE%"=="5" set /p "GTRADE_AR_AXES=    axes (comma-separated): "
REM  Not one of 1-5: use whatever was typed verbatim as the axes (e.g. "qd" or "qd,features").
if not defined GTRADE_AR_AXES set "GTRADE_AR_AXES=%MODE%"

REM  Label for the WHOLE run: the base and every candidate share it, so this is
REM  the setting that decides what an axis is even able to measure. Flat ifs, no
REM  parenthesized block, so each set /p sees the fresh value.
echo.
echo [1a] Label for this run (applies to the base AND every candidate):
echo     1 = direction (default; next-bar label)
echo     2 = triple_barrier, horizon 20 bars
echo     3 = triple_barrier, custom horizon
echo         The weighting axis is a no-op under 1: a next-bar label spans one
echo         bar, the uniqueness weights come out all-ones, and every candidate
echo         equals the base. Pick 2 or 3 when running that axis.
set "LBL=1"
set /p "LBL=    choice [1]: "
set "GTRADE_LABEL_MODE=direction"
set "GTRADE_LABEL_HORIZON=1"
if "%LBL%"=="2" set "GTRADE_LABEL_MODE=triple_barrier"
if "%LBL%"=="2" set "GTRADE_LABEL_HORIZON=20"
if "%LBL%"=="3" set "GTRADE_LABEL_MODE=triple_barrier"
if "%LBL%"=="3" set "GTRADE_LABEL_HORIZON=20"
if "%LBL%"=="3" set /p "GTRADE_LABEL_HORIZON=    horizon in bars [20]: "

echo.
echo [2] Proposer:
echo     1 = evolutionary (no LLM, fully autonomous)
echo     2 = local LLM (Ollama; any installed model - you pick below)
echo     3 = Anthropic API (needs ANTHROPIC_API_KEY)
echo     4 = OpenAI API (needs OPENAI_API_KEY)
set "PROP=1"
set /p "PROP=    choice [1]: "
set "GTRADE_AR_PROPOSER=evolutionary"
set "GTRADE_AR_LLM="

if "%PROP%"=="2" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=ollama"
  echo.
  echo     Installed local models:
  python -m core.llm_proposer --list-ollama
  echo     Enter = auto-detect ^(first gemma, else first installed^).
  set /p "GTRADE_AR_LLM_MODEL=    model name [auto]: "
)

if "%PROP%"=="3" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=anthropic"
  set /p "GTRADE_AR_LLM_MODEL=    Anthropic model [claude-opus-4-8]: "
)

if "%PROP%"=="4" (
  set "GTRADE_AR_PROPOSER=llm"
  set "GTRADE_AR_LLM=openai"
  set /p "GTRADE_AR_LLM_MODEL=    OpenAI model [gpt-4o]: "
)

REM  Token budget for the LLM (proposer + wiki). Reasoning models like gemma spend
REM  tokens on an internal trace before the answer; too small a cap returns EMPTY
REM  content. 0 = no cap (the local model is free; the only cost is wall-clock time).
if not "%GTRADE_AR_PROPOSER%"=="llm" goto :skiptoks
echo.
echo     LLM max tokens  (0 = no cap; gemma reasoning needs room)
set "GTRADE_AR_LLM_MAX_TOKENS=8000"
set /p "GTRADE_AR_LLM_MAX_TOKENS=    max tokens [8000]: "
echo.
echo     Seconds allowed for ONE call. A large local model on CPU needs far
echo     more than the 600s SDK default; a timeout is not retried, so a value
echo     that is too small costs the whole LLM arm for the run. 0 = no limit.
set "GTRADE_AR_LLM_TIMEOUT=3600"
set /p "GTRADE_AR_LLM_TIMEOUT=    timeout seconds [3600]: "
echo.
echo     Reflect before proposing?  The model first writes one line on why the
echo     recent experiments failed, then proposes with that in front of it.
echo     Costs one extra call per step.  1 = off (default)   2 = on
set "REFL=1"
set /p "REFL=    choice [1]: "
set "GTRADE_AR_REFLECT="
if "%REFL%"=="2" set "GTRADE_AR_REFLECT=1"
:skiptoks

echo.
set "AR_BUDGET=15"
set /p "AR_BUDGET=[3] Budget (NEW search iterations this run) [15]: "

echo.
echo [4] Objective (how per-asset held-out lifts are reduced to one number):
echo     1 = mean (average)   2 = min (lift the floor)   3 = median (robust average)
echo     4 = cvar (mean of the worst 25%%)   5 = sharpe (consistency)   6 = trimmed (no extremes)
set "OBJ=1"
set /p "OBJ=    choice [1]: "
set "GTRADE_AR_OBJECTIVE=mean"
if "%OBJ%"=="2" set "GTRADE_AR_OBJECTIVE=min"
if "%OBJ%"=="3" set "GTRADE_AR_OBJECTIVE=median"
if "%OBJ%"=="4" set "GTRADE_AR_OBJECTIVE=cvar"
if "%OBJ%"=="5" set "GTRADE_AR_OBJECTIVE=sharpe"
if "%OBJ%"=="6" set "GTRADE_AR_OBJECTIVE=trimmed_mean"

echo.
echo [4b] Score basis (WHICH number the objective above is applied to):
echo     1 = raw ensemble Score (default)
echo     2 = neural contribution (ensemble minus a CatBoost-only run)
echo         Use 2 to hunt specifically for something that revives the neural
echo         members. The qd SEARCH always runs the CatBoost-only screen, so
echo         basis 2 re-scores the final GATE only; the elites are still picked
echo         by CatBoost alone. That is why earlier neural runs read flat.
echo         Basis 2 is a DIFFERENCE, so an axis that helps both learners equally
echo         (weighting, labeling, folds) reads as zero on it. Use basis 1 there.
echo     3 = neural AUC (Net_AUC: the nets' own probabilities, averaged over ALL
echo         folds). Bases 1 and 2 are both Score, and the Score is a backtest of
echo         discrete signals behind a fold-admission threshold: measured on this
echo         GPU, the SAME config and seed lands 0.45 to 1.52 Score apart, more
echo         than the adoption floor. Basis 3 is a rank statistic on raw
echo         probabilities and does not inherit that. Use 3 for any neural A/B.
echo         Floor is GTRADE_AR_ADOPT_AUC (0.005), not the Score floor, and the
echo         neural_lift veto switches off because basis 3 already measures the
echo         nets. Costs nothing extra: the same single training run.
echo     4 = neural GAIN (Ens_AUC minus CB_AUC: what the ensemble adds over
echo         CatBoost alone). What basis 2 always meant, on a rank statistic
echo         instead of a Score. Use 4 rather than 3 when the nets are given a
echo         target other than direction - there basis 3 measures nothing,
echo         because it scores them against the direction label. Same floor
echo         (GTRADE_AR_ADOPT_AUC) and the same free re-key of one train.
echo     5 = ensemble AUC (Ens_AUC as a level). Use this whenever a candidate
echo         changes BOTH learners at once. There basis 4 would reward simply
echo         damaging CatBoost, because its delta is d(Ens_AUC) - d(CB_AUC) and
echo         anything that hurts CatBoost drives the second term negative.
echo         Basis 5 asks only: did the ensemble improve.
set "BAS=1"
set /p "BAS=    choice [1]: "
set "GTRADE_AR_SCORE_BASIS="
if "%BAS%"=="2" set "GTRADE_AR_SCORE_BASIS=neural"
if "%BAS%"=="3" set "GTRADE_AR_SCORE_BASIS=net_auc"
if "%BAS%"=="4" set "GTRADE_AR_SCORE_BASIS=net_gain"
if "%BAS%"=="5" set "GTRADE_AR_SCORE_BASIS=ens_auc"


echo.
echo [5] Research wiki?  (compounding findings; uses the LLM backend)
echo     1 = off (default)   2 = on
set "WIKI=1"
set /p "WIKI=    choice [1]: "
REM  "0", not empty. `set "VAR="` DELETES the variable in cmd, and auto_research
REM  then calls load_dotenv(), which fills a MISSING key from .env - where
REM  GTRADE_AR_WIKI=1. So the empty form silently ran the wiki (and its gemma4
REM  calls) on every run that answered "off" here. An explicit falsy value is
REM  present in the environment, so load_dotenv leaves it alone.
set "GTRADE_AR_WIKI=0"
if "%WIKI%"=="2" set "GTRADE_AR_WIKI=1"

echo.
echo [6] RL scheduler?  (learned budget allocation over the QD child sources;
echo     Thompson bandit + CMA/novelty emitters; the adoption gate is untouched)
echo     1 = off (default)   2 = on
set "RL=1"
set /p "RL=    choice [1]: "
set "GTRADE_AR_RL="
if "%RL%"=="2" set "GTRADE_AR_RL=1"

echo.
echo ------------------------------------------------------------
echo   axes=%GTRADE_AR_AXES%  label=%GTRADE_LABEL_MODE%/%GTRADE_LABEL_HORIZON%
echo   proposer=%GTRADE_AR_PROPOSER%  llm=%GTRADE_AR_LLM%
echo   model=%GTRADE_AR_LLM_MODEL%  maxtok=%GTRADE_AR_LLM_MAX_TOKENS%  timeout=%GTRADE_AR_LLM_TIMEOUT%
echo   wiki=%GTRADE_AR_WIKI%  reflect=%GTRADE_AR_REFLECT%
echo   budget=%AR_BUDGET%  objective=%GTRADE_AR_OBJECTIVE%  basis=%GTRADE_AR_SCORE_BASIS%  rl=%GTRADE_AR_RL%
echo   heldout=%GTRADE_AR_HELDOUT%  train_seed=%GTRADE_SEED%
echo ------------------------------------------------------------
set "GO=Y"
set /p "GO=Start? [Y/n]: "
if /i "%GO%"=="n" exit /b 0

python auto_research.py

REM  Optional wiki lint (flat, no parenthesized block, so set /p + if see the fresh value).
if not "%GTRADE_AR_WIKI%"=="1" goto :nolint
echo.
set "LINT=n"
set /p "LINT=Lint the research wiki now (reconcile + prune)? [y/N]: "
if /i "%LINT%"=="y" python -c "from dotenv import load_dotenv; load_dotenv(); from core import ar_wiki; ar_wiki.lint_wiki()"
:nolint

echo.
echo Done. Review _auto_research_log.json / _qd_archive.json / _ar_findings.json.
if "%GTRADE_AR_WIKI%"=="1" echo Research wiki: _ar_wiki\*.md  (also on the /research Web UI page).
if "%GTRADE_AR_RL%"=="1" echo RL scheduler state: rl_scheduler_v1.json  (arm posteriors are in the run log above).
pause
goto :end

REM == Re-gate: re-score the best already-found candidates under the current gate ==
:regate
echo.
echo   RE-GATE: re-scores the best already-found candidate genomes (from _qd_archive +
echo   _ar_findings) under the current stronger gate. Reuses past experiments; trains only
echo   the top-K on the held-out set. Adopts nothing - flags winners for you.
set "RGK=8"
set /p "RGK=    top-K candidates [8]: "
echo.
echo     CB-only pre-screen first (cheaper, coarser)?  1 = no (default)   2 = yes
set "RGS=1"
set /p "RGS=    choice [1]: "
set "RGSCREEN="
if "%RGS%"=="2" set "RGSCREEN=--regate-screen"
echo.
echo   Running: auto_research.py --regate --regate-k %RGK% %RGSCREEN%
python auto_research.py --regate --regate-k %RGK% %RGSCREEN%
echo.
echo Done. Review _ar_findings.json (mode=regate) for the new verdicts.
pause

:end
