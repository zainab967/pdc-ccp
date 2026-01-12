"""
Performance Benchmark Script for Heat Equation Solvers
Run this script to collect execution times and generate comparison data.

Usage: python benchmark.py
"""

import subprocess
import re
import csv
import os

# Configuration
RUNS_PER_TEST = 3  # Average over multiple runs
RESULTS_FILE = "benchmark_results.csv"

def run_command(cmd, cwd="."):
    """Run a command and return stdout."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

def extract_time(output):
    """Extract execution time from program output."""
    # Match patterns like "Execution time: 1.234 ms" or "GPU time: 1.234 ms"
    match = re.search(r'(?:Execution|GPU) time:\s+([\d.]+)\s*ms', output)
    if match:
        return float(match.group(1))
    return None

def extract_error(output):
    """Extract L2 error from program output."""
    match = re.search(r'L2 Error:\s+([\d.e+-]+)', output)
    if match:
        return float(match.group(1))
    return None

def benchmark_sequential():
    """Benchmark sequential version."""
    print("Testing Sequential...")
    times = []
    for _ in range(RUNS_PER_TEST):
        output = run_command(["heat_seq.exe"])
        time = extract_time(output)
        if time:
            times.append(time)
    
    if times:
        avg_time = sum(times) / len(times)
        return {"impl": "Sequential", "threads": 1, "time_ms": avg_time}
    return None

def benchmark_openmp(thread_counts=[1, 2, 4, 8]):
    """Benchmark OpenMP with different thread counts."""
    results = []
    for threads in thread_counts:
        print(f"Testing OpenMP with {threads} threads...")
        times = []
        for _ in range(RUNS_PER_TEST):
            output = run_command(["heat_omp.exe", str(threads)])
            time = extract_time(output)
            if time:
                times.append(time)
        
        if times:
            avg_time = sum(times) / len(times)
            results.append({"impl": "OpenMP", "threads": threads, "time_ms": avg_time})
    return results

def benchmark_mpi(process_counts=[1, 2, 4]):
    """Benchmark MPI with different process counts."""
    results = []
    for procs in process_counts:
        print(f"Testing MPI with {procs} processes...")
        times = []
        for _ in range(RUNS_PER_TEST):
            output = run_command(["mpiexec", "-n", str(procs), "heat_mpi.exe"])
            time = extract_time(output)
            if time:
                times.append(time)
        
        if times:
            avg_time = sum(times) / len(times)
            results.append({"impl": "MPI", "threads": procs, "time_ms": avg_time})
    return results

def benchmark_cuda():
    """Benchmark CUDA version."""
    print("Testing CUDA...")
    times = []
    for _ in range(RUNS_PER_TEST):
        output = run_command(["heat_cuda.exe"])
        time = extract_time(output)
        if time:
            times.append(time)
    
    if times:
        avg_time = sum(times) / len(times)
        return {"impl": "CUDA", "threads": "GPU", "time_ms": avg_time}
    return None

def calculate_speedup(results, baseline_time):
    """Calculate speedup relative to baseline."""
    for r in results:
        r["speedup"] = baseline_time / r["time_ms"] if r["time_ms"] > 0 else 0
    return results

def main():
    print("=" * 50)
    print("  Heat Equation Solver Benchmarks")
    print("=" * 50)
    print()
    
    all_results = []
    
    # Sequential baseline
    seq = benchmark_sequential()
    if seq:
        all_results.append(seq)
        baseline_time = seq["time_ms"]
    else:
        print("WARNING: Sequential benchmark failed")
        baseline_time = 1.0
    
    # OpenMP
    omp_results = benchmark_openmp()
    all_results.extend(omp_results)
    
    # MPI (skip if mpiexec not available)
    try:
        mpi_results = benchmark_mpi()
        all_results.extend(mpi_results)
    except:
        print("WARNING: MPI benchmark skipped (mpiexec not found)")
    
    # CUDA
    cuda = benchmark_cuda()
    if cuda:
        all_results.append(cuda)
    
    # Calculate speedups
    all_results = calculate_speedup(all_results, baseline_time)
    
    # Save results to CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "threads", "time_ms", "speedup"])
        writer.writeheader()
        writer.writerows(all_results)
    
    # Print summary
    print()
    print("=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Implementation':<15} {'Threads/Procs':<15} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['impl']:<15} {str(r['threads']):<15} {r['time_ms']:<15.3f} {r.get('speedup', 0):.2f}x")
    
    print()
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
