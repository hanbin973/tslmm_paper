import numpy as np

import tskit
import msprime
import tsinfer
import tsdate

from snakemake.script import snakemake

# load data
ts = tskit.load(snakemake.input.ts_path)

# infer tree sequence
sample_data = tsinfer.SampleData.from_tree_sequence(ts)
inferred_ts = tsinfer.infer(sample_data)
simplified_ts = tsdate.preprocess_ts(inferred_ts)
dated_ts = tsdate.variational_gamma(simplified_ts, mutation_rate=1e-8, rescaling_intervals=10)

# save data
dated_ts.dump(snakemake.output.ts_save_path)