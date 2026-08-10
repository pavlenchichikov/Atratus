@echo off
REM ===========================================================================
REM  ONE definition of the project environment, shared by activate_env.bat and
REM  setup_gpu.bat. It exists because the two used to hardcode different names
REM  ("gtrade_gpu" in the launcher, whatever the machine happened to have in
REM  conda), the launcher hid the mismatch with 2>nul, and training silently ran
REM  on base python with no GPU for months.
REM
REM  Change the name here and both scripts follow.
REM ===========================================================================

REM Conda env that every project script runs in.
set "GTRADE_ENV=jackpot_gpu"

REM Python 3.10 is not a preference, it is the ceiling: TensorFlow 2.10 is the
REM last release with native-Windows CUDA, and 2.10 does not build for 3.11+.
set "GTRADE_PY_VER=3.10"

REM TF range. The upper bound matters: 2.11 and later are CPU-only on Windows.
set "GTRADE_TF_SPEC=tensorflow>=2.10,<2.11"

REM CUDA/cuDNN pairs tried in order. 11.8 covers Turing, Ampere and Ada
REM (RTX 20xx/30xx/40xx); 11.2 is the fallback for older drivers. Cards newer
REM than Ada (compute capability 9.0+) are not supported by CUDA 11.x at all and
REM fall back to CPU - see the README table.
set "GTRADE_CUDA_PRIMARY=cudatoolkit=11.8 cudnn=8.6"
set "GTRADE_CUDA_FALLBACK=cudatoolkit=11.2 cudnn=8.1"
