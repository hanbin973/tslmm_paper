import argparse
import time
import numpy as np
import pandas as pd
import utils

def run_benchmark(n_samples, mu, seq_len, ne, rho, iterations, output_file):
    # Fixed parameters
    seed = 42
    n_threads = 1 # Single-threaded for fair comparison
    
    print(f"--- Benchmarking: n_samples={n_samples}, mu={mu} ---")
    
    # 1. Setup Simulation
    # Note: We do not time this setup phase as we are profiling the matrix operation itself
    ts, arg = utils.simulate_and_prepare_arg(n_samples, seq_len, ne, rho, mu, seed)
    
    # Define random vector v
    np.random.seed(seed)
    v = np.random.normal(size=(n_samples, 1))
    
    # 2. Compute Correlation (Accuracy Check)
    # We run the computation once outside the timing loop to check agreement.
    print("Computing correlation...")
    
    # TSKit: Expected Covariance (Branch Lengths)
    y_ts = ts.genetic_relatedness_vector(v, mode="branch", span_normalise=False, centre=False) * mu
    
    # ARG: Realized Covariance (Genotype Matrix)
    y_arg = utils.compute_arg_Gv(arg, v, n_threads=n_threads)
    
    # Calculate Pearson Correlation
    # We flatten the arrays because they are likely shape (N, 1)
    correlation = np.corrcoef(y_ts.flatten(), y_arg.flatten())[0, 1]
    print(f"  Correlation (TS vs ARG): {correlation:.6f}")

    # 3. Timing Loop
    print(f"Starting {iterations} timing iterations...")
    times_ts = []
    times_arg = []
    
    for i in range(iterations):
        # --- Measure TSKit ---
        start_ts = time.perf_counter()
        _ = ts.genetic_relatedness_vector(v, mode="branch", span_normalise=False, centre=False)
        end_ts = time.perf_counter()
        times_ts.append(end_ts - start_ts)
        
        # --- Measure ARG Needle ---
        start_arg = time.perf_counter()
        _ = utils.compute_arg_Gv(arg, v, n_threads=n_threads)
        end_arg = time.perf_counter()
        times_arg.append(end_arg - start_arg)

    # 4. Compile Results
    results = {
        "n_samples": n_samples,
        "mu": mu,
        "seq_len": seq_len,
        "n_trees": ts.num_trees,
        "n_mutations": ts.num_mutations if hasattr(ts, 'num_mutations') else arg.num_mutations(),
        "correlation": correlation,  # <--- NEW FIELD
        "ts_mean_sec": np.mean(times_ts),
        "ts_std_sec": np.std(times_ts),
        "arg_mean_sec": np.mean(times_arg),
        "arg_std_sec": np.std(times_arg),
        "iterations": iterations
    }
    
    # 5. Save to CSV
    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False)
    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TSKit vs ARG-Needle MatMul")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of haploid samples")
    parser.add_argument("--mu", type=float, required=True, help="Mutation rate")
    parser.add_argument("--seq_len", type=float, default=1e6, help="Sequence length")
    parser.add_argument("--ne", type=float, default=1000, help="Effective population size")
    parser.add_argument("--rho", type=float, default=1e-8, help="Recombination rate")
    parser.add_argument("--iterations", type=int, default=10, help="Number of timing iterations")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    run_benchmark(
        args.n_samples, 
        args.mu, 
        args.seq_len, 
        args.ne, 
        args.rho, 
        args.iterations, 
        args.output
    )