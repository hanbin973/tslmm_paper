import numpy as np

import tskit
import msprime
import demes

from snakemake.script import snakemake

import os

# load data
ts = tskit.load(snakemake.input.ts_path)

# trmca
x = np.zeros(ts.num_samples)
x[0] = 1
tmrca = np.dot(x, ts.genetic_relatedness_vector(x, mode='branch', centre=False))

# save result
with open(snakemake.output.out_path, "w") as output_file:
    print(tmrca, file=output_file)

