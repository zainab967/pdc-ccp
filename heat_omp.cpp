/**
 * OpenMP Parallel Heat Equation Solver
 * 1D Rod Temperature Evolution Simulation
 * 
 * Physics: dT/dt = alpha * d²T/dx² (Heat Equation)
 * Method:  FTCS (Forward-Time Central-Space) Finite Difference
 * Parallelization: OpenMP for spatial loop
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <omp.h>

// ============== SIMULATION PARAMETERS ==============
const double L = 1.0;           // Rod length (meters)
const double ALPHA = 0.01;      // Thermal diffusivity (m²/s)
const int N = 10000;            // Number of spatial points
const double T_FINAL = 0.5;     // Simulation time (seconds)

const double DX = L / (N - 1);  // Spatial step
const double DT = 0.4 * DX * DX / ALPHA;  // Time step (CFL stable)
const double R = ALPHA * DT / (DX * DX);  // Diffusion number

const double PI = 3.14159265358979323846;

// ============== FUNCTIONS ==============

void initialize(std::vector<double>& T) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double x = i * DX;
        T[i] = 100.0 * sin(PI * x);
    }
    T[0] = 0.0;
    T[N-1] = 0.0;
}

double analytical_solution(double x, double t) {
    return 100.0 * exp(-ALPHA * PI * PI * t) * sin(PI * x);
}

double calculate_error(const std::vector<double>& T, double t) {
    double error = 0.0;
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < N; i++) {
        double x = i * DX;
        double diff = T[i] - analytical_solution(x, t);
        error += diff * diff;
    }
    return sqrt(error / N);
}

void save_results(const std::vector<double>& T, double t, const std::string& filename) {
    std::ofstream file(filename);
    file << "# x, T_numerical, T_analytical\n";
    for (int i = 0; i < N; i++) {
        double x = i * DX;
        file << x << ", " << T[i] << ", " << analytical_solution(x, t) << "\n";
    }
    file.close();
}

// ============== MAIN SOLVER ==============
int main(int argc, char* argv[]) {
    // Set number of threads (default: max available)
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    std::cout << "========================================\n";
    std::cout << "  OpenMP Heat Equation Solver\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Parameters:\n";
    std::cout << "  Rod length L    = " << L << " m\n";
    std::cout << "  Grid points N   = " << N << "\n";
    std::cout << "  Spatial step dx = " << DX << " m\n";
    std::cout << "  Time step dt    = " << DT << " s\n";
    std::cout << "  Diffusivity α   = " << ALPHA << " m²/s\n";
    std::cout << "  Diffusion num r = " << R << " (must be <= 0.5)\n";
    std::cout << "  Final time      = " << T_FINAL << " s\n";
    std::cout << "  OpenMP threads  = " << num_threads << "\n\n";
    
    if (R > 0.5) {
        std::cerr << "ERROR: Unstable! r = " << R << " > 0.5\n";
        return 1;
    }
    
    std::vector<double> T_old(N);
    std::vector<double> T_new(N);
    
    initialize(T_old);
    
    int num_steps = static_cast<int>(T_FINAL / DT);
    std::cout << "Running " << num_steps << " time steps...\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        // Parallel update of interior points
        #pragma omp parallel for
        for (int i = 1; i < N - 1; i++) {
            T_new[i] = T_old[i] + R * (T_old[i+1] - 2.0*T_old[i] + T_old[i-1]);
        }
        
        // Boundary conditions (sequential - small overhead)
        T_new[0] = 0.0;
        T_new[N-1] = 0.0;
        
        std::swap(T_old, T_new);
        t += DT;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    double error = calculate_error(T_old, t);
    
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t << " s\n";
    std::cout << "  Execution time:   " << elapsed.count() << " ms\n";
    std::cout << "  L2 Error:         " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << *std::max_element(T_old.begin(), T_old.end()) << " °C\n";
    std::cout << "  Threads used:     " << num_threads << "\n";
    
    save_results(T_old, t, "results_omp.csv");
    std::cout << "\nResults saved to results_omp.csv\n";
    
    return 0;
}
