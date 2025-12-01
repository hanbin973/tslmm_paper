######## snakemake preamble start (automatically inserted, do not edit) ########
import sys;sys.path.extend(['/home/hblee/.conda/envs/jax_main/lib/python3.12/site-packages', '/home/hblee/tslmm_paper/08-infer', '/home/hblee/.conda/envs/jax_main/lib/python3.12', '/home/hblee/.conda/envs/jax_main/lib/python3.12/lib-dynload', '/home/hblee/.conda/envs/jax_main/lib/python3.12/site-packages', '/home/hblee/.cache/snakemake/snakemake/source-cache/runtime-cache/tmpnjpn_kbu/file/home/hblee/tslmm_paper/08-infer', '/home/hblee/tslmm_paper/08-infer']);import pickle;from snakemake import script;script.snakemake = pickle.loads(b'\x80\x04\x95\x1a\x05\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94(\x8c\x1btree_sequences/ts_n50000.ts\x94\x8c$inferred_tree_sequences/ts_n50000.ts\x94e}\x94(\x8c\x06_names\x94}\x94(\x8c\x07ts_path\x94K\x00N\x86\x94\x8c\x08its_path\x94K\x01N\x86\x94u\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh\x15h\x06\x8c\x0eAttributeGuard\x94\x93\x94)\x81\x94}\x94\x8c\x04name\x94h\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbh\x0fh\nh\x11h\x0bub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8c\x14pred/pred_n50000.npy\x94a}\x94(h\r}\x94\x8c\tsave_path\x94K\x00N\x86\x94sh\x13]\x94(h\x15h\x16eh\x15h\x18)\x81\x94}\x94h\x1bh\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbh%h"ub\x8c\r_params_store\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94\x8c\x0550000\x94a}\x94(h\r}\x94\x8c\x0fnum_individuals\x94K\x00N\x86\x94sh\x13]\x94(h\x15h\x16eh\x15h\x18)\x81\x94}\x94h\x1bh\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbh3h0ub\x8c\r_params_types\x94}\x94\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94h0a}\x94(h\r}\x94\x8c\x0fnum_individuals\x94K\x00N\x86\x94sh\x13]\x94(h\x15h\x16eh\x15h\x18)\x81\x94}\x94h\x1bh\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbh3h0ub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01J\xa0\x86\x01\x00M\xba\x03M\xe8\x03M\xba\x03\x8c\x04/tmp\x94\x8c\x08standard\x94\x8c\x06jonth0\x94K\x01\x8c\x04100g\x94K\x04\x8c\x0b02-00:00:01\x94e}\x94(h\r}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94\x8c\x06mem_mb\x94K\x02N\x86\x94\x8c\x07mem_mib\x94K\x03N\x86\x94\x8c\x07disk_mb\x94K\x04N\x86\x94\x8c\x08disk_mib\x94K\x05N\x86\x94\x8c\x06tmpdir\x94K\x06N\x86\x94\x8c\tpartition\x94K\x07N\x86\x94\x8c\x07account\x94K\x08N\x86\x94\x8c\x05nodes\x94K\tN\x86\x94\x8c\x03mem\x94K\nN\x86\x94\x8c\x0bnum_threads\x94K\x0bN\x86\x94\x8c\ntime_limit\x94K\x0cN\x86\x94uh\x13]\x94(h\x15h\x16eh\x15h\x18)\x81\x94}\x94h\x1bh\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbhUK\x01hWK\x01hYJ\xa0\x86\x01\x00h[M\xba\x03h]M\xe8\x03h_M\xba\x03hahN\x8c\tpartition\x94hO\x8c\x07account\x94hP\x8c\x05nodes\x94K\x01\x8c\x03mem\x94hQ\x8c\x0bnum_threads\x94K\x04\x8c\ntime_limit\x94hRub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h\r}\x94h\x13]\x94(h\x15h\x16eh\x15h\x18)\x81\x94}\x94h\x1bh\x15sbh\x16h\x18)\x81\x94}\x94h\x1bh\x16sbub\x8c\x06config\x94}\x94\x8c\x04rule\x94\x8c\x04pred\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8c /home/hblee/tslmm_paper/08-infer\x94ub.');del script;from snakemake.logging import logger;from snakemake.script import snakemake; logger.printshellcmds = True;__real_file__ = __file__; __file__ = '/home/hblee/tslmm_paper/08-infer/04-prediction.py';
######## snakemake preamble end #########
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
    X = rng.normal(size=(tree_sequence.num_individuals, 5)) # covariates
    g = sim_genetic_value(tree_sequence) * np.sqrt(mutation_rate * tau) # genetic value 
    e = rng.normal(size=tree_sequence.num_individuals) * np.sqrt(sigma) # residual
    b = rng.normal(size=5) # fixed effect size
    y = X @ b + g + e # trait value
    return y, X, b, g

# load data
ts = tskit.load(snakemake.input.ts_path)
its = tskit.load(snakemake.input.its_path)

# paramter config
x = np.zeros(its.num_samples)
x[0] = 1
tmrca = np.dot(x, ts.genetic_relatedness_vector(x, mode='branch', centre=False))
tmrca_infer = np.dot(x, its.genetic_relatedness_vector(x, mode='branch', centre=False))

mu = 1 / (4 * tmrca[0])
mu_infer = 1 / (4 * tmrca_infer[0])
rng = np.random.default_rng()

# simulate & fit tslmm
taus = np.linspace(0, 2, 6)
num_simulations = 5
pred = np.empty((6, num_simulations, 3, int(ts.num_individuals/2)))
for i, tau in enumerate(taus):
    varcov = 1, tau
    for j in range(num_simulations):
        traits, covariates, fixef, genetic_values = simulate(*varcov, ts, mu, rng=rng)

        # fit variance component
        subset = np.arange(0, ts.num_individuals, 2)
        lmm_true = TSLMM(ts, mu, traits[subset], covariates[subset],
                    phenotyped_individuals=subset, rng=rng, 
                    num_threads=1)
        lmm_true.fit_variance_components(method='ai', haseman_elston=True, verbose=True)
        lmm_infer = TSLMM(its, mu_infer, traits[subset], covariates[subset],
                    phenotyped_individuals=subset, rng=rng, 
                    num_threads=1)
        lmm_infer.fit_variance_components(method='ai', haseman_elston=True, verbose=True)

        # compute prediction
        blups_true = lmm_true.predict(np.arange(ts.num_individuals), variance_samples=0)
        blups_infer = lmm_infer.predict(np.arange(ts.num_individuals), variance_samples=0)

        # store prediction
        not_subset = np.setdiff1d(np.arange(ts.num_individuals), subset)
        pred[i,j,0,:] = genetic_values[not_subset]
        pred[i,j,1,:] = blups_true[not_subset]
        pred[i,j,2,:] = blups_infer[not_subset]

np.save(snakemake.output.save_path, pred)
