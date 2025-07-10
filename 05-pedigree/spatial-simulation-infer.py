import tskit
import tszip
import tsinfer
import os
import numpy as np
import logging
import argparse
import json


docstring = \
"""
- Infer tree sequence from tszip'd spatial simulation
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
    #parser.add_argument(
    #    "--num-individuals", type=int, default=5000,
    #    help="Number of individuals to subsample for inference",
    #)
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
        filename=args.out_prefix + f".{args.it}.inferred.log",
        level=logging.INFO, 
        filemode="w",
    )
    
    genealogies_path = args.in_prefix + f".{args.it}.genealogies.trees"
    ts = tszip.decompress(genealogies_path)
    logging.info(f"Input tree sequence:\n{ts}")

    sim_mutations = json.loads(ts.provenance(-1).record)["parameters"]
    assert sim_mutations["command"] == "sim_mutations"
    mutation_rate = sim_mutations["rate"]

    
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

    # tsinfer
    inferred_path = args.out_prefix + f".{args.it}.inferred.trees"
    if not os.path.exists(inferred_path) or args.overwrite:
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ts_inf = tsinfer.infer(sample_data, num_threads=args.num_threads)
        tszip.compress(ts_inf, inferred_path)
    else:
        ts_inf = tszip.load(inferred_path)
    
    # tsdate
    dated_path = args.out_prefix + f".{args.it}.inferred.dated.trees"
    if not os.path.exists(dated_path) or args.overwrite:
        import tsdate
        ts_date = tsdate.date(
            tsdate.preprocess_ts(ts_inf),
            mutation_rate=mutation_rate,
            rescaling_intervals=20000,
        )
        tszip.compress(ts_date, dated_path)
    else:
        ts_date = tszip.load(dated_path)
