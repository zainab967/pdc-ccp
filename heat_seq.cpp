/**
 * Sequential Heat Equation Solver
 * 1D Rod Temperature Evolution Simulation
 * 
 * Physics: dT/dt = alpha * d²T/dx² (Heat Equation)
 * Method:  FTCS (Forward-Time Central-Space) Finite Difference
 * 
 * Boundary: T(0,t) = T(L,t) = 0
 * Initial:  T(x,0) = 100 * sin(pi*x)
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>

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

// Initial condition: T(x,0) = 100 * sin(pi*x)
void initialize(std::vector<double>& T) {
    for (int i = 0; i < N; i++) {
        double x = i * DX;
        T[i] = 100.0 * sin(PI * x);
    }
    // Enforce boundary conditions
    T[0] = 0.0;
    T[N-1] = 0.0;
}

// Analytical solution: T(x,t) = 100 * exp(-alpha*pi²*t) * sin(pi*x)
double analytical_solution(double x, double t) {
    return 100.0 * exp(-ALPHA * PI * PI * t) * sin(PI * x);
}

// Calculate L2 error between numerical and analytical
double calculate_error(const std::vector<double>& T, double t) {
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        double x = i * DX;
        double diff = T[i] - analytical_solution(x, t);
        error += diff * diff;
    }
    return sqrt(error / N);
}

// Save temperature profile to file
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
int main() {
    std::cout << "========================================\n";
    std::cout << "  Sequential Heat Equation Solver\n";
    std::cout << "========================================\n\n";
    
    // Print parameters
    std::cout << "Parameters:\n";
    std::cout << "  Rod length L    = " << L << " m\n";
    std::cout << "  Grid points N   = " << N << "\n";
    std::cout << "  Spatial step dx = " << DX << " m\n";
    std::cout << "  Time step dt    = " << DT << " s\n";
    std::cout << "  Diffusivity α   = " << ALPHA << " m²/s\n";
    std::cout << "  Diffusion num r = " << R << " (must be <= 0.5)\n";
    std::cout << "  Final time      = " << T_FINAL << " s\n\n";
    
    // Check stability
    if (R > 0.5) {
        std::cerr << "ERROR: Unstable! r = " << R << " > 0.5\n";
        return 1;
    }
    
    // Allocate arrays
    std::vector<double> T_old(N);
    std::vector<double> T_new(N);
    
    // Initialize
    initialize(T_old);
    
    // Count time steps
    int num_steps = static_cast<int>(T_FINAL / DT);
    
    std::cout << "Running " << num_steps << " time steps...\n\n";
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Time-stepping loop
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        // Update interior points using FTCS scheme
        for (int i = 1; i < N - 1; i++) {
            T_new[i] = T_old[i] + R * (T_old[i+1] - 2.0*T_old[i] + T_old[i-1]);
        }
        
        // Apply boundary conditions
        T_new[0] = 0.0;
        T_new[N-1] = 0.0;
        
        // Swap arrays
        std::swap(T_old, T_new);
        
        t += DT;
    }
    
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    // Calculate final error
    double error = calculate_error(T_old, t);
    
    // Print results
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t << " s\n";
    std::cout << "  Execution time:   " << elapsed.count() << " ms\n";
    std::cout << "  L2 Error:         " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << *std::max_element(T_old.begin(), T_old.end()) << " °C\n";
    
    // Save results
    save_results(T_old, t, "results_seq.csv");
    std::cout << "\nResults saved to results_seq.csv\n";
    
    return 0;
}
