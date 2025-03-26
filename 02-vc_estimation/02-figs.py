import numpy as np
import matplotlib.pyplot as plt

import tskit
import msprime
import demes

from snakemake.script import snakemake

variance_components = np.load(snakemake.input.variance_components)

fig, ax = plt.subplots(figsize=(6,4))
# boxplot
tick_labels = np.linspace(0, 1, 6)
boxplot = ax.boxplot(variance_components.T, tick_labels=tick_labels.round(2))
for xcoord, ycoord in zip(np.arange(tick_labels.size)+1, tick_labels):
    margin = 0.4
    ax.plot([xcoord-margin, xcoord+margin], [ycoord, ycoord], ls='dotted', lw=2, color='red')

# annotation
#ax.set_xlim([-0.01, 2.01])
ax.set_xlabel(r'True $\tau^2$', fontsize=13)
ax.set_ylabel(r'Estimated $\widehat{\tau^2}$', fontsize=13)
for median in boxplot['medians']:
    median.set_color('black')

plt.savefig(snakemake.output.save_path, bbox_inches='tight')
