import msprime
import arg_needle_lib
import numpy as np

def simulate_and_prepare_arg(n_samples, seq_len, Ne, rho, mu, seed=42):
    """
    Simulates ancestry using msprime, converts it to an ARG using arg_needle_lib,
    and populates it with mutations for matrix operations.
    """
    # 1. Simulate Ancestry (Tree Sequence)
    ts = msprime.sim_ancestry(
        samples=n_samples, 
        ploidy=1, 
        population_size=Ne, 
        recombination_rate=rho, 
        sequence_length=seq_len, 
        random_seed=seed
    )
    
    # 2. Convert to ARG & Initialize Topology
    arg = arg_needle_lib.tskit_to_arg(ts)
    arg.populate_children_and_roots()
    
    # 3. Generate Mutations (The 'Z' matrix structure)
    # Using a fixed seed for mutation generation ensures consistency across benchmarks
    arg_needle_lib.generate_mutations(arg, mu=mu, random_seed=123, num_mutations_hint=0)
    arg.populate_mutations_on_edges()
    
    # 4. Prepare Internal Structures for Fast MatMul
    arg_needle_lib.prepare_matmul(arg)
    
    return ts, arg

def compute_arg_Gv(arg, v, n_threads=1):
    """
    Computes Gv = (Z * Z^T) * v using the two-step ARG matrix multiplication.
    Step 1: y* = Z^T * v  (Axis: Mutations)
    Step 2: y  = Z * y* (Axis: Samples)
    """
    # Step 1: y* = Z^T * v 
    # arg_matmul(axis="mutations") computes X * Z^T. We pass v.T to get v^T * Z^T = (Z v)^T.
    y_star = arg_needle_lib.arg_matmul(
        arg, v.T, axis="mutations", standardize=False, diploid=False, n_threads=n_threads
    ).T
    
    # Step 2: y = Z * y*
    y = arg_needle_lib.arg_matmul(
        arg, y_star, axis="samples", standardize=False, diploid=False, n_threads=n_threads
    )
    return y