@echo off
REM ===========================================================================
REM  Creates the project conda environment on a fresh machine.
REM
REM  Run it directly, or let activate_env.bat call it with --auto the first time
REM  a launcher starts on a machine that has no environment yet.
REM
REM  The env name and every version pin come from env_config.bat. Nothing here
REM  hardcodes them: the launcher and the installer naming different envs is the
REM  exact bug that left the GPU unused for months.
REM ===========================================================================
title ATRATUS ENV SETUP
color 0E
call "%~dp0env_config.bat"

set "AUTO="
if /I "%~1"=="--auto" set "AUTO=1"

if not defined AUTO cls
echo =============================================================
echo    ATRATUS ENV SETUP: Auto-Detect + Install
echo =============================================================
echo.
echo  Environment : %GTRADE_ENV%  (Python %GTRADE_PY_VER%)
echo  TensorFlow  : %GTRADE_TF_SPEC%
echo.
echo  Python %GTRADE_PY_VER% and TF 2.10 are not preferences. TensorFlow dropped
echo  native-Windows CUDA support after 2.10, and 2.10 has no build for 3.11+.
echo  Any newer combination sees no GPU on Windows, whatever card is installed.
echo.
echo  GPU support:
echo    - NVIDIA Turing/Ampere/Ada (RTX 20xx/30xx/40xx, GTX 16xx): CUDA 11.8
echo    - older NVIDIA with an older driver: CUDA 11.2 fallback
echo    - anything newer than Ada, or AMD/Intel: CPU only
echo.
echo =============================================================
if not defined AUTO pause

echo.
echo [1/5] Creating conda environment %GTRADE_ENV% (Python %GTRADE_PY_VER%)...
echo -------------------------------------------------------
call conda create -n %GTRADE_ENV% python=%GTRADE_PY_VER% -y
if errorlevel 1 (
    echo [ERROR] Failed to create environment!
    pause
    exit /b 1
)

echo.
echo [2/5] Detecting GPU...
echo -------------------------------------------------------
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    set HAS_GPU=1
) else (
    echo [INFO] No NVIDIA GPU detected. Will install CPU-only TensorFlow.
    set HAS_GPU=0
)

echo.
echo [3/5] Installing CUDA toolkit + cuDNN...
echo -------------------------------------------------------
if "%HAS_GPU%"=="1" (
    call conda install -n %GTRADE_ENV% -c conda-forge %GTRADE_CUDA_PRIMARY% -y
    if errorlevel 1 (
        echo [WARN] %GTRADE_CUDA_PRIMARY% failed, trying %GTRADE_CUDA_FALLBACK%...
        call conda install -n %GTRADE_ENV% -c conda-forge %GTRADE_CUDA_FALLBACK% -y
    )
) else (
    echo [SKIP] No GPU - skipping CUDA installation.
)

echo.
echo [4/5] Installing TensorFlow + project dependencies...
echo -------------------------------------------------------
if "%HAS_GPU%"=="1" (
    call conda run -n %GTRADE_ENV% pip install "%GTRADE_TF_SPEC%" --no-cache-dir
) else (
    call conda run -n %GTRADE_ENV% pip install "tensorflow-cpu>=2.10,<2.11" --no-cache-dir
)
if errorlevel 1 (
    echo [ERROR] TensorFlow installation failed!
    pause
    exit /b 1
)

call conda run -n %GTRADE_ENV% pip install -r requirements.txt --no-cache-dir

echo.
echo [5/5] Verifying GPU setup...
echo =============================================================
call conda run -n %GTRADE_ENV% python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(); print(f'  TensorFlow {tf.__version__}'); print(f'  GPU devices: {len(gpus)}'); [print(f'    - {g.name}') for g in gpus]; print(f'  Status: {\"GPU READY\" if gpus else \"CPU ONLY\"}')"

echo.
echo =============================================================
echo    SETUP COMPLETE
echo =============================================================
echo.
echo  Environment: %GTRADE_ENV%
echo  Activate:    call activate_env.bat
echo  Run:         run_gtrade.bat  /  auto_research.bat
echo.
if "%HAS_GPU%"=="0" (
    echo  NOTE: Running on CPU. Training works but takes many times longer.
    echo.
)
echo  Models trained here are NOT interchangeable with models trained under a
echo  Keras 3 environment: Keras 2 writes HDF5 into the .keras name, Keras 3
echo  writes a zip, and neither reads the other. Retrain after switching.
echo.
if not defined AUTO pause
