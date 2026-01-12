/**
 * MPI Parallel Heat Equation Solver
 * 1D Rod Temperature Evolution Simulation
 * 
 * Physics: dT/dt = alpha * d²T/dx² (Heat Equation)
 * Method:  FTCS (Forward-Time Central-Space) Finite Difference
 * Parallelization: Domain decomposition with ghost cell exchange
 * 
 * Run: mpiexec -n 4 heat_mpi.exe
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <mpi.h>

// ============== SIMULATION PARAMETERS ==============
const double L = 1.0;           // Rod length (meters)
const double ALPHA = 0.01;      // Thermal diffusivity (m²/s)
const int N = 10000;            // Total number of spatial points
const double T_FINAL = 0.5;     // Simulation time (seconds)

const double DX = L / (N - 1);  // Spatial step
const double DT = 0.4 * DX * DX / ALPHA;  // Time step (CFL stable)
const double R = ALPHA * DT / (DX * DX);  // Diffusion number

const double PI = 3.14159265358979323846;

// ============== FUNCTIONS ==============

double initial_condition(double x) {
    return 100.0 * sin(PI * x);
}

double analytical_solution(double x, double t) {
    return 100.0 * exp(-ALPHA * PI * PI * t) * sin(PI * x);
}

// ============== MAIN SOLVER ==============
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate local domain size
    int local_n = N / size;
    int remainder = N % size;
    
    // Handle uneven distribution
    int start_idx = rank * local_n + std::min(rank, remainder);
    if (rank < remainder) local_n++;
    int end_idx = start_idx + local_n - 1;
    
    // Allocate local arrays with ghost cells
    // ghost[0] | local_data[1..local_n] | ghost[local_n+1]
    int array_size = local_n + 2;  // +2 for ghost cells
    std::vector<double> T_old(array_size);
    std::vector<double> T_new(array_size);
    
    // Initialize local domain
    for (int i = 1; i <= local_n; i++) {
        int global_i = start_idx + i - 1;
        double x = global_i * DX;
        T_old[i] = initial_condition(x);
    }
    
    // Apply global boundary conditions
    if (rank == 0) T_old[1] = 0.0;
    if (rank == size - 1) T_old[local_n] = 0.0;
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  MPI Heat Equation Solver\n";
        std::cout << "========================================\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  Rod length L    = " << L << " m\n";
        std::cout << "  Grid points N   = " << N << "\n";
        std::cout << "  MPI processes   = " << size << "\n";
        std::cout << "  Diffusion num r = " << R << " (must be <= 0.5)\n\n";
    }
    
    int num_steps = static_cast<int>(T_FINAL / DT);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        // Exchange ghost cells with neighbors
        int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
        
        // Send right, receive left
        MPI_Sendrecv(&T_old[local_n], 1, MPI_DOUBLE, right_neighbor, 0,
                     &T_old[0], 1, MPI_DOUBLE, left_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Send left, receive right
        MPI_Sendrecv(&T_old[1], 1, MPI_DOUBLE, left_neighbor, 1,
                     &T_old[local_n + 1], 1, MPI_DOUBLE, right_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Update interior points
        for (int i = 1; i <= local_n; i++) {
            // Skip global boundaries
            int global_i = start_idx + i - 1;
            if (global_i == 0 || global_i == N - 1) {
                T_new[i] = 0.0;
            } else {
                T_new[i] = T_old[i] + R * (T_old[i+1] - 2.0*T_old[i] + T_old[i-1]);
            }
        }
        
        std::swap(T_old, T_new);
        t += DT;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    // Calculate local error
    double local_error = 0.0;
    for (int i = 1; i <= local_n; i++) {
        int global_i = start_idx + i - 1;
        double x = global_i * DX;
        double diff = T_old[i] - analytical_solution(x, t);
        local_error += diff * diff;
    }
    
    // Reduce to get global error
    double global_error;
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Gather results on rank 0
    std::vector<double> T_global;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        T_global.resize(N);
    }
    
    // Gather local sizes and displacements
    int local_count = local_n;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    // Gather all data (skip ghost cells)
    MPI_Gatherv(&T_old[1], local_n, MPI_DOUBLE,
                T_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  RESULTS\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Final time:       " << t << " s\n";
        std::cout << "  Execution time:   " << (end_time - start_time) * 1000 << " ms\n";
        std::cout << "  L2 Error:         " << std::scientific << sqrt(global_error / N) << "\n";
        std::cout << "  MPI processes:    " << size << "\n";
        
        // Save results
        std::ofstream file("results_mpi.csv");
        file << "# x, T_numerical, T_analytical\n";
        for (int i = 0; i < N; i++) {
            double x = i * DX;
            file << x << ", " << T_global[i] << ", " << analytical_solution(x, t) << "\n";
        }
        file.close();
        std::cout << "\nResults saved to results_mpi.csv\n";
    }
    
    MPI_Finalize();
    return 0;
}
