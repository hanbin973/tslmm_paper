import numpy as np

from snakemake.script import snakemake

npys = [np.load(path) for path in snakemake.input.npy_paths]
arr = np.vstack(npys)

np.save(snakemake.output[0], arr)
