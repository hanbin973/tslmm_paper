import subprocess
import tskit
import tstrait
import msprime
import pickle
import tszip
import numpy as np
import os
import argparse
import warnings

import matplotlib.pyplot as plt


docstring = \
r"""
- Simulate ARG under spatial model with local density regulation using SLiM
- Write out pedigree information from tracked ancestral individuals into msprime.PedigreeBuilder
- Simulate conditional on pedigree with msprime
- Recapitate with a more-or-less arbitrary population size
- Simulate neutral mutations on top of genealogies
- Assign effect size to mutations with tstrait
- Do some sanity checks (e.g. fine-scale spatial clustering of close relatives, phenotypes)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(docstring)
    parser.add_argument(
        "--seed", type=int, default=1024,
        help="Random seed",
    )
    parser.add_argument(
        "--spatial-kernel", type=float, default=0.25,
        help="StdDev of dispersal/interaction/mate choice distance "
        "if this is too small, population will go extinct",
    )
    parser.add_argument(
        "--carrying-cap", type=int, default=5,
        help="Local carrying capacity, "
        "if this is too small, population will go extinct",
    )
    parser.add_argument(
        "--spatial-extent", type=float, default=100.0, #default=500.0,
        help="Width of the square spatial domain",
    )
    parser.add_argument(
        "--sequence-length", type=float, default=1e8,
        help="Sequence length in bp",
    )
    parser.add_argument(
        "--recombination-rate", type=float, default=1e-8,
        help="Recombination rate",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=1e-8,
        help="Mutation rate",
    )
    parser.add_argument(
        "--prop-causal-mutations", type=int, default=0.01,
        help="Proportion of mutations that are causual",
    )
    parser.add_argument(
        "--heritability", type=float, default=1.0,
        help="Narrow sense heritability",
    )
    parser.add_argument(
        "--ticks", type=int, default=200,
        help="Number of ticks to run pedigree simulation",
    )
    parser.add_argument(
        "--burn-in", type=int, default=50,
        help="Number of ticks to discard from start of pedigree simulation",
    )
    parser.add_argument(
        "--in-prefix", type=str, default="spatial-simulation",
        help="Input path for tree sequence",
    )
    parser.add_argument(
        "--out-prefix", type=str, default="spatial-simulation",
        help="Input path for tree sequence",
    )
    parser.add_argument(
        "--it", type=str, default="0",
        help="iteration no.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite all output",
    )
    parser.add_argument(
        "--overwrite-from-msp", action="store_true",
        help="Overwrite output from msprime simulation onwards",
    )
    args = parser.parse_args()
    logfile = open(args.out_prefix + ".log", "w")
    
# msprime simulation conditional on pedigree
    slim_pedigree_path = args.in_prefix + ".pedigree.trees"
    slim_ts = tszip.decompress(slim_pedigree_path)
    pedigree_tables_path = args.in_prefix + ".pedigree.tables"
    pedigree_tables = tszip.decompress(pedigree_tables_path).dump_tables()
    genealogies_path = args.out_prefix + f".{args.it}.genealogies.trees"
    if not os.path.exists(genealogies_path) or args.overwrite or args.overwrite_from_msp:
        # TODO: use a better pop size estimate, or adjust for burn-in
        pedigree_tables.sequence_length = args.sequence_length
        pedigree_ts = msprime.sim_ancestry(
            initial_state=pedigree_tables.tree_sequence(),
            recombination_rate=args.recombination_rate,
            model="fixed_pedigree",
            random_seed=args.seed + 1000,
        )

        # recapitate (complete the genealogies) using census population size,
        # e.g. this is a historically panmixic population that suddenly
        # experiences spatial structure
        population_size = np.bincount(
            slim_ts.nodes_time.astype(int), 
            minlength=args.ticks + 1,
        )[args.ticks - args.burn_in]
        logfile.write(f"Recapitating with population size {population_size}\n")
        logfile.write(
            f"Nodes, edges before recapitating: "
            f"{pedigree_ts.num_nodes}, {pedigree_ts.num_edges}\n"
        )
        full_ts = msprime.sim_ancestry(
            initial_state=pedigree_ts,
            recombination_rate=args.recombination_rate,
            population_size=population_size,
            random_seed=args.seed + 2000,
        )
        logfile.write(
            f"Nodes, edges after recapitating: "
            f"{full_ts.num_nodes}, {full_ts.num_edges}\n"
        )
        full_ts = msprime.sim_mutations(
            full_ts, 
            rate=args.mutation_rate,
            random_seed=args.seed + 3000,
        )
        logfile.write(
            f"Simulated {full_ts.num_mutations} mutations with "
            f"mu={args.mutation_rate}\n"
        )
        logfile.write(
            f"Diversity: {full_ts.diversity():.4f}\n"
            f"Segregating sites: {full_ts.segregating_sites():.4f}\n"
        )
        tszip.compress(full_ts, genealogies_path)
    else:
        full_ts = tszip.decompress(genealogies_path)


    # trait simulation with tstrait
    phenotype_path = args.out_prefix + f".{args.it}.phenotypes.pkl"
    if not os.path.exists(phenotype_path) or args.overwrite or args.overwrite_from_msp:
        num_causal = int(args.prop_causal_mutations * full_ts.num_mutations)
        phenotypes = tstrait.sim_phenotype(
            full_ts, 
            model=tstrait.trait_model(distribution="normal", mean=0, var=1),
            num_causal=num_causal,
            h2=args.heritability,
            random_seed=args.seed + 4000,
        )
        logfile.write(f"Simulated {num_causal} effect sizes\n")
        pickle.dump(phenotypes, open(phenotype_path, "wb"))
    else:
        phenotypes = pickle.load(open(phenotype_path, "rb"))

    # NB: presumably the phenotypes aren't correct for internal individuals in
    # the pedigree, as only a partial sequence is recorded for these


