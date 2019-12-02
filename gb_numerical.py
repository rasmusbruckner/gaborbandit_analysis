""" This script demonstrates learning in A1 using the analytical solution and the numerical approximation

    1. Systematically compare analytical and numerical solution
    2. Illustration of numerical solution
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from gb_simulation import gb_simulation
from latex_plt import latex_plt
# Update matplotlib to use Latex and to change some defaults
import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Plot properties
fontsize = 8

# Create figure
f = plt.figure()
ax_0 = plt.gca()

# -----------------------------------------------------------------
# 1. Systematically compare analytical and numerical solution:
# Systematically compare evolution of learned contingency parameter
# between analytical and numerical solution
# -----------------------------------------------------------------

# Reset random number generator
np.random.seed(123)

# Simulation parameters
T = 40
B = 1
sigma = 0.03
beta = 100
agent = 1

# Run simulation with analytical solution
df_subj_a1 = gb_simulation(T, B, sigma, agent, beta, eval_ana=True)
df_subj_a1['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Plot contingency parameter
for _, group in df_subj_a1.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_0, legend=False, color='red', linewidth=4, alpha=1)

# Reset random number generator
np.random.seed(123)

# Run simulation with numerical solution
df_subj_a1 = gb_simulation(T, B, sigma, agent, beta, eval_ana=False)
df_subj_a1['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Plot contingency parameter
for _, group in df_subj_a1.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_0, legend=False, color='green', linewidth=1, alpha=1)
ax_0.set_ylim([0.2, 1])
ax_0.tick_params(labelsize=fontsize)
ax_0.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
ax_0.set_ylabel(r'Expected Value ($E_{\mu}$)', fontsize=fontsize)
ax_0.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_0.axhline(0.8, color='black', lw=0.5, linestyle='--')
plt.tight_layout()
sns.despine()

# ----------------------------------------------------------------------------------------------------------------------
# 2. Illustration of numerical solution:
# Illustrate that learned contingency parameter converges towards 0.8 with numerical approximation to analytical
# solution. Here we use the numerical approximation because the normalization step in the analytical solution
# requires very small coefficients after a large amount of trials. In the numerical solution, we simply normalize using
# numerical integration
# ----------------------------------------------------------------------------------------------------------------------

# Create figure
f = plt.figure()
ax_0 = plt.gca()

# Simulation parameters
T = 1000
B = 10
sigma = 0.03
beta = 100  # beta parameter of softmax choice rule
agent = 1

# Run simulation
df_subj_a1 = gb_simulation(T, B, sigma, agent, beta, eval_ana=False)
df_subj_a1['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean contingency parameter
mean_corr_a1 = df_subj_a1.groupby(df_subj_a1['t'])['e_mu_t'].mean()

# Plot all contingency parameters
for _, group in df_subj_a1.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_0, legend=False, color='gray', linewidth=1, alpha=1)
ax_0.set_ylim([0.2, 1])
ax_0.tick_params(labelsize=fontsize)
ax_0.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
ax_0.set_ylabel(r'Expected Value ($E_{\mu}$)', fontsize=fontsize)
x = np.linspace(1, T, T)
ax_0.plot(x, mean_corr_a1, linewidth=4, color='green', alpha=1)
ax_0.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_0.axhline(0.8, color='black', lw=0.5, linestyle='--')

# Show plots
plt.show()
