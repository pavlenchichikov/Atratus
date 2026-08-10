@echo off
REM ===========================================================================
REM  Activates the project conda environment, creating it first if this machine
REM  has never run the project. Called by every launcher; safe to call directly.
REM
REM  Portability notes, all of them learned the hard way:
REM
REM  1. "conda activate" does not exist in a bare cmd.exe unless conda has been
REM     initialised for it. A double-clicked .bat therefore printed "The system
REM     cannot find the path specified" and carried on with whatever python was
REM     on PATH. The conda hook is called explicitly below.
REM  2. That failure does NOT set errorlevel, so the result is verified through
REM     CONDA_PREFIX instead of the exit code.
REM  3. Conda lives in different places depending on the installer. Every common
REM     location is probed, plus CONDA_EXE when the shell already exported it.
REM  4. TensorFlow dropped native-Windows CUDA after 2.10, so the environment is
REM     pinned to Python 3.10 + TF 2.10. Base anaconda (3.11+) sees no GPU at
REM     all, whatever card is installed.
REM ===========================================================================
call "%~dp0env_config.bat"

REM -- locate the conda installation ------------------------------------------
REM %%~fI normalises the path: without it CONDA_EXE yields "...\Scripts\.." and
REM every later comparison against it fails on a literal string mismatch.
set "CONDA_ROOT="
if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
if defined CONDA_ROOT if not exist "%CONDA_ROOT%\Scripts\activate.bat" set "CONDA_ROOT="

for %%D in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\Continuum\anaconda3"
    "%ProgramData%\anaconda3"
    "%ProgramData%\Anaconda3"
    "%ProgramData%\miniconda3"
    "C:\anaconda3"
    "C:\miniconda3"
) do (
    if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
)

if not defined CONDA_ROOT (
    echo.
    echo [ERROR] No conda installation found.
    echo         Install Miniconda ^(https://docs.conda.io/en/latest/miniconda.html^)
    echo         and run this again. Without it TensorFlow runs on CPU and a full
    echo         training pass takes many times longer.
    pause
    exit /b 1
)

REM -- create the environment on first use ------------------------------------
if not exist "%CONDA_ROOT%\envs\%GTRADE_ENV%\python.exe" (
    echo.
    echo [env] "%GTRADE_ENV%" not found on this machine. Creating it now.
    echo       This downloads CUDA, cuDNN and TensorFlow and takes a while.
    call "%~dp0setup_gpu.bat" --auto
    if not exist "%CONDA_ROOT%\envs\%GTRADE_ENV%\python.exe" (
        echo [ERROR] Setup did not produce the environment. See the output above.
        pause
        exit /b 1
    )
)

REM -- bring conda into this cmd session, then switch to the project env ------
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate %GTRADE_ENV%

REM -- verify by RESULT, not by exit code -------------------------------------
REM Checked by substance rather than by comparing two path strings: conda roots
REM arrive spelled differently depending on how the shell was started, and a
REM cosmetic mismatch must not abort a correctly activated environment.
set "ENV_OK="
if defined CONDA_PREFIX for %%I in ("%CONDA_PREFIX%") do (
    if /I "%%~nxI"=="%GTRADE_ENV%" if exist "%CONDA_PREFIX%\python.exe" set "ENV_OK=1"
)
if not defined ENV_OK (
    echo.
    echo [ERROR] Environment "%GTRADE_ENV%" is NOT active.
    echo         CONDA_PREFIX = %CONDA_PREFIX%
    echo         Check the name with:  conda env list
    echo         Refusing to continue: on base python TensorFlow sees no GPU and
    echo         a full run would take days instead of hours.
    pause
    exit /b 1
)

echo [env] %GTRADE_ENV% active  ^(%CONDA_PREFIX%^)
REM Explicit success: conda's own hooks leave a stale errorlevel behind, and a
REM caller that checks it would otherwise treat a good activation as a failure.
exit /b 0
