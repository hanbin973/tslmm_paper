import numpy as np

import tskit
import msprime
import demes

from snakemake.script import snakemake

import sys
sys.path.append("/home/hblee/tslmm")
from tslmm.tslmm import TSLMM
from tslmm.simulations import sim_genetic_value

def simulate(sigma, tau, tree_sequence, mutation_rate, rng=None, subset=None, center_covariance=False):
    """
    float sigma: non-genetic variance, i.e., residual
    float tau: genetic variance
    tskit.TreeSequence tree_sequence: tree sequence
    float mutation_rate: mutation rate (e.g. 1e-8)
    np.random.Generator rng: numpy random number generator
    np.ndarray subset: index of individuals to use # currently not active option
    bool center_covariance: center covariance if True
    """
    if rng is None: rng = np.random.default_rng()
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    X = rng.normal(size=(tree_sequence.num_individuals, 4)) # covariates
    X = np.hstack((np.ones((tree_sequence.num_individuals, 1)), X)) # add intercept
    g = sim_genetic_value(tree_sequence) * np.sqrt(mutation_rate * tau) # genetic value 
    e = rng.normal(size=tree_sequence.num_individuals) * np.sqrt(sigma) # residual
    b = rng.normal(size=5) # fixed effect size
    y = X @ b + g + e # trait value
    return y, X, b, g

# load data
ts = tskit.load(snakemake.input.ts_path)

# paramter config
x = np.zeros(ts.num_samples)
x[0] = 1
tmrca = np.dot(x, ts.genetic_relatedness_vector(x, mode='branch', centre=False))

mu = 1 / (4 * tmrca[0])
rng = np.random.default_rng()

# simulate & fit tslmm
subset_est = np.empty((1, 2))
notsubset_est = np.empty((1, 2))
varcov = 1, 3
subset = np.arange(0, ts.num_individuals, 2)
not_subset = np.setdiff1d(np.arange(ts.num_individuals), subset)
traits, covariates, fixef, genetic_values = simulate(*varcov, ts, mu, rng=rng)
preconditioner_rank = min(1000, int(ts.num_individuals / 50))

lmm = TSLMM(ts, mu, traits[subset], covariates[subset], rng=rng, phenotyped_individuals=subset, centre=True)
lmm.fit_variance_components(method='ai', haseman_elston=True, verbose=True)
subset_est[0] = lmm.variance_components
lmm = TSLMM(ts, mu, traits[not_subset], covariates[not_subset], rng=rng, phenotyped_individuals=not_subset, centre=True)
lmm.fit_variance_components(method='ai', haseman_elston=True, verbose=True)
notsubset_est[0] = lmm.variance_components

np.save(snakemake.output.save_path[0], subset_est)
np.save(snakemake.output.save_path[1], notsubset_est)
