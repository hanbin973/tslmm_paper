import numpy

import tskit
import msprime
import demes

from snakemake.script import snakemake

# define demography
demography = msprime.Demography()
demography.add_population(name="Panmictic", initial_size=100_000, growth_rate=0.01)
num_individuals = int(snakemake.params.num_individuals)
seq_length = float(snakemake.params.seq_length)
ts = msprime.sim_ancestry(samples={"Panmictic": num_individuals},
                          sequence_length=seq_length,
                          recombination_rate=1e-8,
                          demography=demography,
                          ploidy=2,
                          random_seed=1)
ts = msprime.sim_mutations(ts,
                           rate=1e-8,
                           random_seed=1)
ts.dump(snakemake.output.ts_save_path)
