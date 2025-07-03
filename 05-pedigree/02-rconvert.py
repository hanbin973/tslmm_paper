import tskit
import tszip
import numpy as np
import argparse
import pickle

docstring = \
        """
        - Parse tstrait output to txt
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(docstring)
    parser.add_argument(
            "--in-prefix", type=str, default="spatial-simulation",
            help="Input prefix for tree sequences, etc",
            )
    parser.add_argument(
            "--out-prefix", type=str, default="spatial-simulation",
            help="Output prefix for tree sequences, etc",
            )
    parser.add_argument(
            "--it", type=str, default="0",
            help="Count of simulation",
            )
    args = parser.parse_args()
    
    genealogies_path = args.in_prefix + f".{args.it}.genealogies.trees"
    pheno_path = args.in_prefix + f".{args.it}.phenotypes.pkl"

    ts = tszip.decompress(genealogies_path)
    with open(pheno_path, 'rb') as file:    
        tstrait_out = pickle.load(file)
    samples, = np.where(ts.individuals_time == 0)
    pheno = tstrait_out.phenotype.phenotype[samples].values[::-1]

    # tree sequence surgery - keep only contemporary individuals
    tables = ts.dump_tables()
    ind_tables = tables.individuals
    ind_tables.replace_with(ind_tables[samples])
    ind_tables.parents = np.full(2 * len(samples), -1, dtype=np.int32)

    node_tables = tables.nodes
    nodes_individual = node_tables.individual
    nodes_historic, = np.where(nodes_individual >= len(samples))
    nodes_individual[nodes_historic] = -1
    node_tables.individual = nodes_individual

    _ts = tables.tree_sequence()
    ts = _ts.simplify(_ts.individuals_nodes.ravel())

    # scale trait
    x = np.zeros(ts.num_samples); x[0] = 1
    tmrca = np.dot(x, ts.genetic_relatedness_vector(x, mode="branch", centre=False))
    scale = .5 / np.sqrt(tmrca * 1e-10)
    pheno = pheno * scale
    
    out_path = args.out_prefix + f".{args.it}.phenotypes.txt"
    np.savetxt(out_path, pheno)
