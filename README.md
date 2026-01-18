# PDC-CCP: Heat Equation Parallel Solver

**Complex Computing Problem - Parallel and Distributed Computing (CS 4172)**  
University of Management and Technology, Lahore | Fall 2025

## 📋 Problem Statement

Simulate the temperature evolution `T(x, t)` of a 1D rod using the heat equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

**Conditions:**
- Rod length: L = 1 meter
- Boundary: T(0, t) = T(L, t) = 0°C
- Initial: T(x, 0) = 100 × sin(πx)
- Thermal diffusivity: α = 0.01 m²/s   assumed as a constant because the material was unspecified.

## 🚀 Implementations

| File | Method | Description |
|------|--------|-------------|
| `heat_seq.cpp` | Sequential | Baseline CPU implementation |
| `heat_omp.cpp` | OpenMP | Shared memory parallelism |
| `heat_mpi.cpp` | MPI | Distributed memory parallelism |
| `heat_cuda.cu` | CUDA | GPU acceleration |

## 📊 Performance Results (N=10,000)

| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| Sequential | 5,365 | 1.00x |
| **OpenMP (4T)** | **3,415** | **1.57x** |
| MPI (4P) | 7,610 | 0.70x |
| CUDA | 11,572 | 0.46x |

**Key Finding:** OpenMP achieves best performance for this iterative problem. MPI/CUDA show overhead limitations due to 1.25M kernel/message operations.

## 🛠️ Build Instructions

### Prerequisites
- Visual Studio 2022 with C++ tools
- MS-MPI SDK and Runtime (for MPI)
- CUDA Toolkit 12.x (for CUDA)

### Build Commands (VS Developer Command Prompt)

```batch
# Navigate to project
cd D:\Repositories\PDC-CCP

# Build Sequential
cl /EHsc /O2 heat_seq.cpp /Fe:heat_seq.exe

# Build OpenMP
cl /EHsc /O2 /openmp heat_omp.cpp /Fe:heat_omp.exe

# Build MPI
cl /EHsc /O2 /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" heat_mpi.cpp /Fe:heat_mpi.exe /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86" msmpi.lib

# Build CUDA (from regular PowerShell)
nvcc -O2 heat_cuda.cu -o heat_cuda.exe
```

## ▶️ Run Instructions

```batch
# Sequential
heat_seq.exe

# OpenMP (4 threads)
heat_omp.exe 4

# MPI (4 processes)
mpiexec -n 4 heat_mpi.exe

# CUDA
heat_cuda.exe
```

## 📈 Benchmarking

```batch
# Run benchmarks and collect data
python benchmark.py

# Generate performance charts
python generate_graphs.py
```

## 📁 Project Structure

```
PDC-CCP/
├── heat_seq.cpp          # Sequential implementation
├── heat_omp.cpp          # OpenMP implementation
├── heat_mpi.cpp          # MPI implementation
├── heat_cuda.cu          # CUDA implementation
├── benchmark.py          # Automated benchmarking
├── generate_graphs.py    # Chart generation
├── build.bat             # Build script
├── report/
│   └── main.tex          # LaTeX report (IEEE format)
└── *.png                 # Performance charts
```

## 📄 Report

- **Overleaf:** [View Report](https://www.overleaf.com/read/hyqzgndjpvww#669d84)

## 🔬 Algorithm

**FTCS (Forward-Time Central-Space)** finite difference:

```
T[i]^(n+1) = T[i]^n + r × (T[i+1]^n - 2×T[i]^n + T[i-1]^n)
```

Where `r = α×Δt/Δx²` (must be ≤ 0.5 for stability)

## 👥 Authors

- **syed shaheer nasir**  - f2022266454@umt.edu.pk
- **Zainab Usman** - f2022266828@umt.edu.pk

Department of Computer Science  
University of Management and Technology, Lahore  

