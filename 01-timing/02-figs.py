import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tskit
import msprime
import demes

from snakemake.script import snakemake

times = pd.concat([pd.DataFrame(np.load(time_path)) for time_path in snakemake.input.runtimes], axis=0)

fig, ax = plt.subplots(1, 3, figsize=(4*3, 4), sharey=True)

# individual
gby_mean = times.groupby(1)[0].mean()
ax[0].plot(gby_mean.index, gby_mean)
ax[0].set_xlabel('Number of individuals', fontsize=12)
ax[0].set_ylabel('Time (s)', fontsize=12)

gby_mean = times.groupby(2)[0].mean()
ax[1].plot(gby_mean.index, gby_mean)
ax[1].set_xlabel('Number of nodes', fontsize=12)

gby_mean = times.groupby(3)[0].mean()
ax[2].plot(gby_mean.index, gby_mean)
ax[2].set_xlabel('Number of edges', fontsize=12)

for i in range(3):
    ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.tight_layout()
plt.savefig(snakemake.output.save_path, bbox_inches='tight')