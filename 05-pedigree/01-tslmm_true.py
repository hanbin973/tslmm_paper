import tskit
import tszip
import numpy as np
import logging
import argparse
import pickle

import sys
sys.path.append('/home/hblee/tslmm')
import tslmm.tslmm as tslmm 

docstring = \
"""
- Run tslmm
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
    
    # load files
    ts = tszip.decompress(genealogies_path)
    with open(pheno_path, "rb") as _file:
        _pheno = pickle.load(_file)
    samples, = np.where(ts.individuals_time == 0)
    pheno = _pheno.phenotype.phenotype[samples].values
    covariates = np.ones((len(samples), 1))

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
    pheno = pheno * scale + np.random.normal(0, 1, size=pheno.shape[0])

    # run tslmm
    mu, rng = .25 / tmrca, np.random.default_rng()
    lmm = tslmm.TSLMM(ts, mu, pheno, covariates, rng=rng, centre=True)
    lmm.fit_variance_components(method='ai', haseman_elston=True, verbose=True) 

    # get blup
    blup_path = args.out_prefix + f".{args.it}.blup.npy"
    blups, _ = lmm.predict(np.arange(ts.num_individuals), variance_samples=5) 
    np.save(blup_path, blups)




