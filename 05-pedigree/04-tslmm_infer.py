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
            "--tree-prefix", type=str, default="spatial-simulation",
            help="Input prefix for tree sequences, etc",
            )
    parser.add_argument(
            "--pheno-prefix", type=str, default="spatial-simulation",
            help="Input prefix for phenotypes",
            )
    parser.add_argument(
            "--it", type=str, default="0",
            help="Count of simulation",
            )
    parser.add_argument(
            "--out-prefix", type=str, default="blup_infer/",
            help="Path to save result",
            )
    args = parser.parse_args()
    
    genealogies_path = args.tree_prefix + f".{args.it}.inferred.dated.trees"
    pheno_path = args.pheno_prefix + f".{args.it}.phenotypes.pkl"

    logging.basicConfig(
            filename=args.out_prefix + f".{args.it}.blup.log",
            level=logging.INFO,
            filemode="w",
            )
    
    # load files
    ts = tszip.decompress(genealogies_path)
    with open(pheno_path, "rb") as _file:
        _pheno = pickle.load(_file)
    samples, = np.where(ts.individuals_time == 0)
    pheno = _pheno.phenotype.phenotype[samples].values
    covariates = np.ones((len(samples), 1))

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




