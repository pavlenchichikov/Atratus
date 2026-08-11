@echo off
REM ===========================================================================
REM  Runs one command inside the project GPU environment.
REM
REM  Usage (from a launcher, note the nested cmd - that is the point):
REM      cmd /c ""%~dp0run_in_env.bat" python ab_build.py --run"
REM
REM  The OUTER pair of quotes is required, not decoration: when the string after
REM  /c starts with a quote, cmd strips the first and the last one, so the single
REM  quoted form loses its argument boundaries and fails with "is not recognized
REM  as an internal or external command".
REM
REM  WHY the nested cmd: "conda activate" changes the CURRENT cmd session and
REM  keeps it changed. Activating inline inside a menu would leave every later
REM  menu item in the GPU environment, and serving there silently loses every
REM  neural champion until the full retrain lands (Keras 2 cannot read the
REM  Keras 3 weights). A child process gets the environment; the menu does not.
REM ===========================================================================
call "%~dp0activate_env.bat"
if errorlevel 1 exit /b 1

%*
