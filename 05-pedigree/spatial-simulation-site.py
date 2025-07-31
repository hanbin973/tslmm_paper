import tskit
import tszip
import tsinfer
import os
import numpy as np
import logging
import argparse
import json
import arg_needle_lib


docstring = \
"""
- Write normalized GRM
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(docstring)
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
        "--num-threads", type=int, default=32,
        help="Number of threads for inference",
    )
    
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite all output",
    )
    parser.add_argument(
        "--overwrite-from-tsdate", action="store_true",
        help="Overwrite output from tsdate step onwards",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.out_prefix + ".{args.it}.site.grm.log",
        level=logging.INFO, 
        filemode="w",
    )
    
    genealogies_path = args.in_prefix + f".{args.it}.genealogies.trees"
    ts = tszip.decompress(genealogies_path)
    logging.info(f"Input tree sequence:\n{ts}")

    # remove internal pedigree individuals, as tsinfer will choke
    sampled_individuals = \
        np.unique(ts.nodes_individual[list(ts.samples())])
    filter_individuals = np.full(ts.num_individuals, False)
    filter_individuals[sampled_individuals] = True
    tab = ts.dump_tables()
    tab.individuals.packset_parents([[]] * ts.num_individuals)
    individual_map = tab.individuals.keep_rows(filter_individuals)
    tab.nodes.individual = individual_map[tab.nodes.individual]
    tab.sort()
    ts = tab.tree_sequence()
    logging.info(f"Pruned tree sequence:\n{ts}")

    grm_path = args.out_prefix + f".{args.it}.grm.txt"
    if not os.path.exists(grm_path) or args.overwite:
        arg = arg_needle_lib.tskit_to_arg(ts)
        arg.populate_children_and_roots()
        grm = arg_needle_lib.monte_carlo_arg_grm(arg, monte_carlo_mu=1e-8)    
        np.savetxt(grm_path, grm)
