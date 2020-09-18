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
from gb_plot_utils import label_subplots

# Update matplotlib to use Latex and to change some defaults
import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

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
ax_0.set_xlabel(r'Trial $(t)$')
ax_0.set_ylabel('Contingency parameter')
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

# -----------------
# 1. Prepare figure
# -----------------

# Define colors
blue_1 = '#46b3e6'
blue_2 = '#4d80e4'
blue_3 = '#2e279d'
green_1 = '#94ed88'
green_2 = '#52d681'
green_3 = '#00ad7c'

# Create figure with multiple subplots
f = plt.figure(figsize=(6.4, 4.8))
ax_00 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)

# Simulation parameters
T = 1000  # number of trials
B = 10  # number of blocks
sigma = 0.04  # perceptual sensitivity
beta = 100  # beta parameter of softmax choice rule

# Simulate data with agent 1
agent = 1
df_subj_a1 = gb_simulation(T, B, sigma, agent, beta, eval_ana=False)
df_subj_a1['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean contingency parameter
mean_corr_a1 = df_subj_a1.groupby(df_subj_a1['t'])['e_mu_t'].mean()

# Plot all contingency parameters
for _, group in df_subj_a1.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_00, legend=False, color='gray', linewidth=1, alpha=1)
ax_00.set_ylim([0.2, 1])
ax_00.set_xlabel('Trial')
ax_00.set_ylabel('Contingency parameter')
x = np.linspace(1, T, T)
ax_00.plot(x, mean_corr_a1, linewidth=4, color=green_1, alpha=1)
ax_00.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_00.axhline(0.8, color='black', lw=0.5, linestyle='--')
ax_00.set_title('Belief-state Bayesian agent (A1)')

# Create next subplot
ax_01 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)

# Simulate data with agent 2
agent = 2
df_subj_a2 = gb_simulation(T, B, sigma, agent, beta, eval_ana=False)
df_subj_a2['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean contingency parameter
mean_corr_a2 = df_subj_a2.groupby(df_subj_a2['t'])['e_mu_t'].mean()

# Plot all contingency parameters
for _, group in df_subj_a2.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_01, legend=False, color='gray', linewidth=1, alpha=1)
ax_01.set_ylim([0.2, 1])
ax_01.set_xlabel('Trial')
ax_01.set_ylabel('Contingency parameter')
x = np.linspace(1, T, T)
ax_01.plot(x, mean_corr_a2, linewidth=4, color=green_2, alpha=1)
ax_01.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_01.axhline(0.8, color='black', lw=0.5, linestyle='--')
ax_01.set_title('Categorical Bayesian agent (A2)')

# Create subplot grid
ax_10 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)

# Simulate data with agent 4
agent = 4
df_subj_a4 = gb_simulation(T, B, sigma, agent, beta)
df_subj_a4['t'] = df_subj_a4['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean contingency parameter
mean_corr_a4 = df_subj_a4.groupby(df_subj_a4['t'])['e_mu_t'].mean()

# Plot all contingency parameters
for _, group in df_subj_a4.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_10, legend=False, color='gray', linewidth=1, alpha=1)
ax_10.set_ylim([0.2, 1])
ax_10.set_xlabel('Trial')
ax_10.set_ylabel('Contingency parameter')
x = np.linspace(1, T, T)
ax_10.plot(x, mean_corr_a4, linewidth=4, color=blue_1, alpha=1)
ax_10.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_10.axhline(0.8, color='black', lw=0.5, linestyle='--')
ax_10.set_title('Belief-state Q-learning agent (A4)')

# Create subplot grid
ax_11 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)

# Simulate data with agent 5
agent = 5
df_subj_a5 = gb_simulation(T, B, sigma, agent, beta)
df_subj_a5['t'] = df_subj_a4['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean contingency parameter
mean_corr_a5 = df_subj_a5.groupby(df_subj_a5['t'])['e_mu_t'].mean()

# Plot all contingency parameters
for _, group in df_subj_a5.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_11, legend=False, color='gray', linewidth=1, alpha=1)
ax_11.set_ylim([0.2, 1])
ax_11.set_xlabel('Trial')
ax_11.set_ylabel('Contingency parameter')
x = np.linspace(1, T, T)
ax_11.plot(x, mean_corr_a5, linewidth=4, color=blue_2, alpha=1)
ax_11.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_11.axhline(0.8, color='black', lw=0.5, linestyle='--')
ax_11.set_title('Categorical Q-learning agent (A5)')
sns.despine()

# Adjust plot properties
plt.subplots_adjust(hspace=0.4, wspace=0.5)

# Label letters
texts = ['a', 'b', 'c', 'd', 'e', 'f']

# Add labels
label_subplots(f, texts)

# Save figure
savename = 'gb_figures/gb_sm_figure_5.pdf'
plt.savefig(savename, dpi=400, transparent=True)

# Show plots
plt.show()
