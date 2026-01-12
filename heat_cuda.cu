/**
 * CUDA Parallel Heat Equation Solver
 * 1D Rod Temperature Evolution Simulation
 * 
 * Physics: dT/dt = alpha * d²T/dx² (Heat Equation)
 * Method:  FTCS (Forward-Time Central-Space) Finite Difference
 * Parallelization: GPU kernel - one thread per grid point
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// ============== SIMULATION PARAMETERS ==============
#define L 1.0               // Rod length (meters)
#define ALPHA 0.01          // Thermal diffusivity (m²/s)
#define N 10000             // Number of spatial points
#define T_FINAL 0.5         // Simulation time (seconds)

#define DX (L / (N - 1))
#define DT (0.4 * DX * DX / ALPHA)
#define R (ALPHA * DT / (DX * DX))

#define PI 3.14159265358979323846

#define BLOCK_SIZE 256

// ============== CUDA ERROR CHECKING ==============
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============== CUDA KERNELS ==============

__global__ void initialize_kernel(double* T, int n, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = i * dx;
        T[i] = 100.0 * sin(PI * x);
    }
    // Boundaries
    if (i == 0) T[0] = 0.0;
    if (i == n - 1) T[n - 1] = 0.0;
}

__global__ void heat_step_kernel(double* T_old, double* T_new, int n, double r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < n - 1) {
        T_new[i] = T_old[i] + r * (T_old[i+1] - 2.0*T_old[i] + T_old[i-1]);
    }
    // Boundary conditions
    if (i == 0) T_new[0] = 0.0;
    if (i == n - 1) T_new[n - 1] = 0.0;
}

// ============== HOST FUNCTIONS ==============

double analytical_solution(double x, double t) {
    return 100.0 * exp(-ALPHA * PI * PI * t) * sin(PI * x);
}

double calculate_error(double* T, double t, int n) {
    double error = 0.0;
    for (int i = 0; i < n; i++) {
        double x = i * DX;
        double diff = T[i] - analytical_solution(x, t);
        error += diff * diff;
    }
    return sqrt(error / n);
}

void save_results(double* T, double t, int n, const char* filename) {
    std::ofstream file(filename);
    file << "# x, T_numerical, T_analytical\n";
    for (int i = 0; i < n; i++) {
        double x = i * DX;
        file << x << ", " << T[i] << ", " << analytical_solution(x, t) << "\n";
    }
    file.close();
}

// ============== MAIN ==============
int main() {
    std::cout << "========================================\n";
    std::cout << "  CUDA Heat Equation Solver\n";
    std::cout << "========================================\n\n";
    
    // Get GPU info
    int deviceId;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    std::cout << "GPU: " << props.name << "\n\n";
    
    std::cout << "Parameters:\n";
    std::cout << "  Rod length L    = " << L << " m\n";
    std::cout << "  Grid points N   = " << N << "\n";
    std::cout << "  Spatial step dx = " << DX << " m\n";
    std::cout << "  Time step dt    = " << DT << " s\n";
    std::cout << "  Diffusivity α   = " << ALPHA << " m²/s\n";
    std::cout << "  Diffusion num r = " << R << " (must be <= 0.5)\n";
    std::cout << "  Block size      = " << BLOCK_SIZE << "\n\n";
    
    if (R > 0.5) {
        std::cerr << "ERROR: Unstable! r = " << R << " > 0.5\n";
        return 1;
    }
    
    // Allocate host memory
    double* h_T = new double[N];
    
    // Allocate device memory
    double *d_T_old, *d_T_new;
    CUDA_CHECK(cudaMalloc(&d_T_old, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T_new, N * sizeof(double)));
    
    // Calculate grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Initialize on GPU
    initialize_kernel<<<numBlocks, BLOCK_SIZE>>>(d_T_old, N, DX);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int num_steps = static_cast<int>(T_FINAL / DT);
    std::cout << "Running " << num_steps << " time steps on GPU...\n\n";
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        heat_step_kernel<<<numBlocks, BLOCK_SIZE>>>(d_T_old, d_T_new, N, R);
        
        // Swap pointers
        double* temp = d_T_old;
        d_T_old = d_T_new;
        d_T_new = temp;
        
        t += DT;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_T, d_T_old, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    double error = calculate_error(h_T, t, N);
    
    // Find max temperature
    double max_temp = 0.0;
    for (int i = 0; i < N; i++) {
        if (h_T[i] > max_temp) max_temp = h_T[i];
    }
    
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t << " s\n";
    std::cout << "  GPU time:         " << elapsed_ms << " ms\n";
    std::cout << "  L2 Error:         " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << max_temp << " °C\n";
    
    save_results(h_T, t, N, "results_cuda.csv");
    std::cout << "\nResults saved to results_cuda.csv\n";
    
    // Cleanup
    delete[] h_T;
    CUDA_CHECK(cudaFree(d_T_old));
    CUDA_CHECK(cudaFree(d_T_new));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
