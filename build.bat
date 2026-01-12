@echo off
echo ========================================
echo   PDC-CCP Heat Equation Solver Build
echo ========================================
echo.

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo [1/4] Building Sequential version...
cl /EHsc /O2 heat_seq.cpp /Fe:heat_seq.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo       SUCCESS) else (echo       FAILED)

echo [2/4] Building OpenMP version...
cl /EHsc /O2 /openmp heat_omp.cpp /Fe:heat_omp.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo       SUCCESS) else (echo       FAILED)

echo [3/4] Building MPI version...
REM Requires Microsoft MPI SDK installed
cl /EHsc /O2 /I"%MSMPI_INC%" heat_mpi.cpp /Fe:heat_mpi.exe /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo       SUCCESS) else (echo       FAILED - Install MS-MPI SDK)

echo [4/4] Building CUDA version...
nvcc -O2 heat_cuda.cu -o heat_cuda.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo       SUCCESS) else (echo       FAILED - Check CUDA installation)

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
