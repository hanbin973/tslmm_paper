import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import statsmodels.formula.api as smf

fname = "runtime_data.csv"
try:
    # this is allowing us to adjust the plotting outside of snakemake
    from snakemake.script import snakemake
    times = pd.concat([pd.DataFrame(np.load(time_path)) for time_path in snakemake.input.runtimes], axis=0)
    np.savetxt(fname, times, delimiter=",", comments="",
               fmt=["%f", "%d", "%d", "%d"],
               header=",".join(["time", "num_individuals", "num_nodes", "num_edges"]))
except:
    print("Not re-generating CSV file!")
    pass

try:
    outpath = snakemake.output.save_path
except:
    outpath = "figs/runtime.pdf"

times = pd.read_csv(fname)
times['num_individuals'] /= 1e3
times['num_edges'] /= 1e6
times['num_nodes'] /= 1e6

times['log_edges'] = np.log(times['num_edges'])
times['log_indivs'] = np.log(times['num_individuals'])
times['log_nodes'] = np.log(times['num_nodes'])
times['log_time'] = np.log(times['time'])

# the main fit
fit = smf.ols(formula='time ~ num_individuals + num_edges : np.log(num_individuals)', data=times).fit()

#print(fit.summary())
#print(fit.params)

# fits used to make smooth lines for the univariate plots
edge_fit = smf.ols(formula='log_edges ~ log_indivs + I(log_indivs**2)', data=times).fit()
indiv_fit = smf.ols(formula='log_indivs ~ log_edges + I(log_edges**2)', data=times).fit()
node_indiv_fit = smf.ols(formula='log_indivs ~ log_nodes + I(log_nodes**2)', data=times).fit()
node_edge_fit = smf.ols(formula='log_edges ~ log_nodes + I(log_nodes**2)', data=times).fit()

fig, ax = plt.subplots(1, 3, figsize=(2*3, 2.2), sharey=True)

# individual
gby_mean = times.groupby('num_individuals')['time'].mean()
pred = pd.DataFrame({
        'num_individuals' : np.linspace(times['num_individuals'].min(), times['num_individuals'].max(), 101),
    })
pred['log_indivs'] = np.log(pred['num_individuals'])
pred['num_edges'] = np.exp(edge_fit.predict(pred))
pred['time'] = fit.predict(pred)

fs=9

ax[0].scatter(gby_mean.index, gby_mean, label='runtime', c='blue')
ax[0].scatter(times['num_individuals'], times['time'], marker='.', c='blue')
ax[0].plot(pred['num_individuals'], pred['time'], c='red', label='model')
ax[0].set_xlabel('Thousands of individuals', fontsize=fs)
ax[0].set_ylabel('Time (s)', fontsize=fs)
# ax[0].legend()

# edge
gby_mean = times.groupby('num_edges')['time'].mean()
pred = pd.DataFrame({
        'num_edges' : np.linspace(times['num_edges'].min(), times['num_edges'].max(), 101),
    })
pred['log_edges'] = np.log(pred['num_edges'])
pred['num_individuals'] = np.exp(indiv_fit.predict(pred))
pred['time'] = fit.predict(pred)

ax[1].scatter(gby_mean.index, gby_mean, label='runtime', c='blue')
ax[1].scatter(times['num_edges'], times['time'], marker='.', c='blue')
ax[1].plot(pred['num_edges'], pred['time'], c='red', label='model')
ax[1].set_xlabel('Millions of edges', fontsize=fs)
ax[1].set_ylabel('Time (s)', fontsize=fs)
# ax[1].legend()

# node
gby_mean = times.groupby('num_nodes')['time'].mean()
pred = pd.DataFrame({
        'num_nodes' : np.linspace(times['num_nodes'].min(), times['num_nodes'].max(), 101),
    })
pred['log_nodes'] = np.log(pred['num_nodes'])
pred['num_individuals'] = np.exp(node_indiv_fit.predict(pred))
pred['num_edges'] = np.exp(node_edge_fit.predict(pred))
pred['time'] = fit.predict(pred)

ax[2].scatter(gby_mean.index, gby_mean, label='runtime', c='blue')
ax[2].scatter(times['num_nodes'], times['time'], marker='.', c='blue')
ax[2].plot(pred['num_nodes'], pred['time'], c='red', label='model')
ax[2].set_xlabel('Millions of nodes', fontsize=fs)
ax[2].set_ylabel('Time (s)', fontsize=fs)
# ax[2].legend()

# some magic formatter suggested by gemini
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

for i in range(3):
    #ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[i].xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig(outpath, bbox_inches='tight')
