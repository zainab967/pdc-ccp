@echo off
setlocal enableextensions
echo ========================================
echo   PDC-CCP Heat Equation Solver Build
echo ========================================
echo.

REM --- Detect Visual Studio (MSVC) environment via vswhere ---
set "VSWHERE_X86=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSWHERE_X64=C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSWHERE="
if exist "%VSWHERE_X86%" set "VSWHERE=%VSWHERE_X86%"
if exist "%VSWHERE_X64%" set "VSWHERE=%VSWHERE_X64%"

set "VCVARS_BAT="
if defined VSWHERE for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS_BAT=%%~I\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS_BAT if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS_BAT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if defined VCVARS_BAT (
  call "%VCVARS_BAT%" >nul 2>&1
) else (
  echo [MSVC] Visual Studio C++ tools not detected.
  echo        Install with: winget install Microsoft.VisualStudio.2022.BuildTools ^^
  echo        --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
)

REM --- Detect Microsoft MPI include/lib paths ---
if not defined MSMPI_INC (
  if exist "C:\Program Files\Microsoft MPI\Inc" set "MSMPI_INC=C:\Program Files\Microsoft MPI\Inc"
  if not defined MSMPI_INC if exist "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
)
if not defined MSMPI_LIB64 (
  if exist "C:\Program Files\Microsoft MPI\Lib\x64" set "MSMPI_LIB64=C:\Program Files\Microsoft MPI\Lib\x64"
  if not defined MSMPI_LIB64 if exist "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
)

REM --- Detect CUDA toolkit (nvcc) ---
if not defined CUDA_PATH (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
  if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
)
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"

REM --- Build targets ---
echo [1/4] Building Sequential version...
cl /EHsc /O2 heat_seq.cpp /Fe:heat_seq.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo       SUCCESS
) else (
  echo       FAILED - MSVC environment missing or compile error
)

echo [2/4] Building OpenMP version...
cl /EHsc /O2 /openmp heat_omp.cpp /Fe:heat_omp.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo       SUCCESS
) else (
  echo       FAILED - MSVC/OpenMP not available
)

echo [3/4] Building MPI version...
if defined MSMPI_INC if defined MSMPI_LIB64 (
  cl /EHsc /O2 /I"%MSMPI_INC%" heat_mpi.cpp /Fe:heat_mpi.exe /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    echo       SUCCESS
  ) else (
    echo       FAILED - Check MS-MPI SDK/Runtime
  )
) else (
  echo       SKIPPED - MS-MPI not detected (install Microsoft.msmpi and Microsoft.msmpisdk)
)

echo [4/4] Building CUDA version...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  nvcc -O2 heat_cuda.cu -o heat_cuda.exe >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    echo       SUCCESS
  ) else (
    echo       FAILED - nvcc compile error
  )
) else (
  echo       SKIPPED - CUDA toolkit not detected (install Nvidia.CUDA via winget)
)

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo Usage:
echo   heat_seq.exe          - Run sequential
echo   heat_omp.exe [threads]- Run OpenMP (e.g., heat_omp.exe 4)
echo   mpiexec -n 4 heat_mpi.exe - Run MPI with 4 processes
echo   heat_cuda.exe         - Run CUDA
echo.
endlocal
