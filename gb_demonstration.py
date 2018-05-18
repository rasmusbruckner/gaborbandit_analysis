""" This script demonstrates the agent models

    1. Figure 3
    2. Figure 4
    3. Additional demonstration of A4 (sampling model)
    4. SM Figure 7

"""

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
from GbAgentVars import AgentVars
from GbAgent import Agent
from gb_simulation import gb_simulation
from truncate_colormap import truncate_colormap


# Use Latex for matplotlib
pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "axes.titlesize": 14,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 12,
    "pgf.rcfonts": False,
    "text.latex.unicode": True,
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
}
matplotlib.rcParams.update(pgf_with_latex)

# Plot properties
markersize = 4
fontsize = 8

# Set perceptual sensitivity conditions
hs = 0.02
ms = 0.04
ls = 0.06

# Set random number generator for reproducible results
np.random.seed(123)

# 1. Figure 3
# -----------

# Call AgentVars and Agent objects
agent_vars = AgentVars()
agent_vars.agent = 1
agent = Agent(agent_vars)  # brauche ich das?

# Set maximal contrast difference value
kappa = 0.08

# Determine x-axis range
x_lim = kappa * 2

# Create list with all contrast differences
d_t = np.linspace(-kappa, kappa, 24)
d_t = [d_t[i:i+1] for i in range(0, len(d_t), 1)]

# Create list with all contrast differences
U = np.linspace(-x_lim, x_lim, 48)
U = [U[i:i+1] for i in range(0, len(U), 1)]

# Define colormap
cmap = truncate_colormap(plt.get_cmap('bone'), 0, 0.7)

# Create range of colors for range of contrast differences
cval_ind = range(len(d_t))
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

# Set resolution for distributions over observatiosn
x = np.linspace(-x_lim, x_lim, 1000)

# Initialize belief state arrays
pi1_hs = np.zeros(24)  # state 1 low noise

# Create figure with multiple subplots
fig = plt.figure(figsize=(10, 3))
ax_0 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=2)
ax_1 = plt.subplot2grid((2, 4), (0, 1), colspan=1, rowspan=2)
ax_2 = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=2)
ax_3 = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)
ax_4 = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1)

# Adjust figure space
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.10, right=0.9, hspace=0.25, wspace=0.35)

# Add figure labels
fig.text(0.04, 0.95, "A)", horizontalalignment='left', verticalalignment='center')
fig.text(0.265, 0.95, "B)", horizontalalignment='left', verticalalignment='center')
fig.text(0.585, 0.95, "C)", horizontalalignment='left', verticalalignment='center')

# Remove unlabeled axes
sns.despine()


# Plot belief over being in s_t = 1
# ---------------------------------
def bs_plot(current_ax, color_ind, current_sigma):
    """ This function plots pi_1 as a function of c_t

    :param current_ax: Current axis
    :param color_ind: cval_ind
    :param current_sigma: Current perceptual sensitivity
    :return:
    """

    # Initialize counter for colors
    counter = 0

    # Cycle over observations
    for i in range(0, len(U)):

        # Current color
        if i < 12:
            color = scalar_map.to_rgba(color_ind[0])
        elif i >= 36:
            color = scalar_map.to_rgba(color_ind[-1])
        else:
            color = scalar_map.to_rgba(color_ind[counter])
            counter += 1

        # Compute and plot belief state for high sensitivity condition
        agent.sigma = current_sigma
        _, pi1 = agent.p_s_giv_o(U[i])
        current_ax.plot(U[i], pi1, '.', color=color)

    return current_ax


ax_0 = bs_plot(ax_0, cval_ind, ms)

# Adjust properties of the plots
ax_0.set_ylim(-0.1, 1.1)
ax_0.tick_params(labelsize=fontsize)
ax_0.set_xlim(-x_lim, x_lim)
ax_0.set_xlabel(r'$c_t$')
ax_0.set_ylabel(r'$\pi_1$', rotation=0, labelpad=10)


# Plot conditional expected value
# -------------------------------

# Initialize belief state arrays
pi_0 = np.linspace(0, 1, 24)
pi_1 = 1 - pi_0

# Set expected value to 1
E_mu_t = 1

# Cycle over observations
for i in range(0, len(pi_1)):

    # Action valence evaluation
    v_a_0 = (pi_0[i] - pi_1[i]) * E_mu_t + pi_1[i]
    v_a_1 = (pi_1[i] - pi_0[i]) * E_mu_t + pi_0[i]
    ax_1.plot(pi_1[i], v_a_0, '.', ms=8, color=(0.6, 0, 0))
    ax_1.plot(pi_1[i], v_a_1, '.', ms=8, color=(0, 0, 0.54))

# Adjust properties of the plots
ax_1.set_ylim(-0.1, 1.1)
ax_1.tick_params(labelsize=fontsize)
ax_1.set_xlabel(r'$\pi_1$')
ax_1.legend([r'$p^{a_t=0}(r_t=1)$', '$p^{a_t=1}(r_t=1)$'])

# Plot softmax choice rule
# ------------------------

# Set inverse temperature parameter to 8
agent.beta = 8

# Initialize conditional expected value and choice probability arrays
p_a_t = np.full([24, 2], np.nan)
v_a_t = np.full([24, 2], np.nan)
v_a_t[:, 1] = np.linspace(0, 1, 24)
v_a_t[:, 0] = 1 - v_a_t[:, 1]

# Cycle over conditional expected values
for i in range(0, len(v_a_t)):

    # Choice probability computation
    agent.v_a_t = v_a_t[i]
    agent.softmax()
    ax_2.plot(v_a_t[i, 0], agent.p_a_t[0], '.', ms=10, color=(0, 0, 0.54))

# Adjust properties of the plots
ax_2.set_ylim(-0.1, 1.1)
ax_2.set_ylabel(r'$p^{\beta}(a_t=1|...)$')
ax_2.set_xlabel(r'$p(r_t=1|a_t=1)$')
plt.tick_params(labelsize=fontsize)


# Plot evolution of p(\mu) for two different constant belief states
# -----------------------------------------------------------------

# Initialize update coefficients
def update_coeff(trial, coeff, q_zero, q_one):
    """ This function updates the polynomial coefficients

    :param trial: Degree of resulting polynomial
    :param coeff: Polynomial coefficient
    :param q_zero: q_0
    :param q_one: q_1
    :return: coeff
    """

    # Todo: Put this function in GbAgent.py

    # Initialize update coefficients
    a = np.zeros([trial])

    # Evaluate last element of a_t
    a[-1] = q_zero * coeff[-1]

    # Evaluate d_1,... (if existent)
    for n in range(0, (trial-2)):
        a[-n-2] = q_one * coeff[-n-1] + q_zero * coeff[-n-2]

    # Evaluate first element of d_t
    a[0] = q_one * coeff[0]

    # Update the coefficients
    coeff = a

    return coeff


# Use sampling agent for illustration
agent.agent = 4

# Define colormap
cmap = truncate_colormap(plt.get_cmap('bone'), 0, 0.7)

# Create range of colors for range of contrast differences
cval_ind = range(5)
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)


def plot_pmu(agent_obj, bs0, bs1, color_ind, s_map, current_ax):
    """ This function illustrates evolution of p(\mu) for given constant belief states

    :param agent_obj: Agent object instance
    :param bs0: p(s_t=0|c_t)
    :param bs1: p(s_t=1|c_t)
    :param color_ind: cval_ind
    :param s_map: scalar_map
    :param current_ax: Current axis
    :return: agent_obj, current_ax
    """
    coeff = np.array([1])
    mu = np.linspace(0, 1, 101)
    p_mu = np.full([101, 5], np.nan)
    p_mu[:, 0] = np.polyval(np.array([1]), mu)

    # Plot analytical solution
    current_ax.plot(mu, p_mu, color='k')
    agent_obj.c_t = coeff

    for i in range(1, 5):

        agent_obj.a_t = np.float(0)

        q0, q1 = agent_obj.compute_q(np.float(1), np.float(bs0), np.float(bs1))

        t = coeff.size + 1
        coeff = update_coeff(t, agent.c_t, q0, q1)
        agent_obj.c_t = coeff
        p_mu[:, i] = np.polyval(agent.c_t, mu)

        current_ax.plot(mu, p_mu[:, i], color=s_map.to_rgba(color_ind[i]))

    return agent_obj, current_ax


# Example of \pi_0 = 1, \pi_1 = 0
agent, ax_3 = plot_pmu(agent, 1, 0, cval_ind, scalar_map, ax_3)

# Adjust properties of the plots
ax_3.set_ylim([-0.1, 5.1])
ax_3.set_ylabel(r'$p_t(\mu)$')
ax_3.set_xlabel(r'$\mu$')

# # Example of \pi_0 = 0.6, \pi_1 = 0.4
agent, ax_4 = plot_pmu(agent, 0.6, 0.4, cval_ind, scalar_map, ax_4)

# Adjust properties of the plots
ax_4.set_ylim([-0.1, 5.1])
ax_4.set_ylabel(r'$p_t(\mu)$')
ax_4.set_xlabel(r'$\mu$')

plt.savefig('model_cartoon.pdf', bbox_inches='tight', transparent=True)


# 1. Figure 4
# -----------

# Create figure with multiple subplots
fig = plt.figure(figsize=(8, 12))
ax_00 = plt.subplot2grid((11, 4), (0, 0))
ax_01 = plt.subplot2grid((11, 4), (0, 1))
ax_02 = plt.subplot2grid((11, 4), (1, 0), rowspan=2)
ax_03 = plt.subplot2grid((11, 4), (1, 1), rowspan=2)
ax_04 = plt.subplot2grid((11, 4), (3, 0), rowspan=2)
ax_05 = plt.subplot2grid((11, 4), (3, 1), rowspan=2)
ax_06 = plt.subplot2grid((11, 4), (5, 0), rowspan=3, colspan=2)
ax_07 = plt.subplot2grid((11, 4), (8, 0), rowspan=3, colspan=2)
ax_10 = plt.subplot2grid((11, 4), (0, 2), colspan=2)
ax_11 = plt.subplot2grid((11, 4), (1, 2), colspan=2)
ax_12 = plt.subplot2grid((11, 4), (2, 2), colspan=2)
ax_13 = plt.subplot2grid((11, 4), (3, 2), colspan=2)
ax_14 = plt.subplot2grid((11, 4), (4, 2), colspan=2)
ax_15 = plt.subplot2grid((11, 4), (5, 2), colspan=2)
ax_16 = plt.subplot2grid((11, 4), (6, 2), colspan=2)
ax_17 = plt.subplot2grid((11, 4), (7, 2), colspan=2)
ax_18 = plt.subplot2grid((11, 4), (8, 2), rowspan=3, colspan=2)

# Define colormap
cmap = truncate_colormap(plt.get_cmap('bone'), 0, 0.7)

# Create range of colors for range of contrast differences
cval_ind = range(len(d_t))
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

# Set resolution for distributions over observatiosn
x = np.linspace(-x_lim, x_lim, 1000)

# Initialize belief state arrays
pi1_hs = np.zeros(24)  # state 1 low noise
pi1_ms = np.zeros(24)  # state 1 high noise

# Plot contrast differences and observation probabilities
# -------------------------------------------------------
for i in range(0, len(d_t)):

    # Plot contrast differences
    # ---------------------------

    # Current color
    cval = scalar_map.to_rgba(cval_ind[i])

    # Low noise
    _, stemlines, _ = ax_00.stem(d_t[i], [1], markerfmt=" ")
    plt.setp(stemlines, linewidth=1, color=cval)  # set stem width and color

    # High noise
    _, stemlines, _ = ax_01.stem(d_t[i], [1], markerfmt=" ")
    plt.setp(stemlines, linewidth=1, color=cval)  # set stem width and color

    # Plot distributions over the observations
    # ----------------------------------------

    # High sensitivity
    fit = stats.norm.pdf(x, d_t[i], hs)
    ax_02.plot(x, fit, '-', linewidth=1, color=cval)

    # Medium sensitivity
    fit = stats.norm.pdf(x, d_t[i], ms)
    ax_03.plot(x, fit, '-', linewidth=1, color=cval)

# Adjust contrast difference plots
ax_00.set_ylim(0, 1)
ax_00.tick_params(labelsize=fontsize)
ax_00.set_xlim(-x_lim, x_lim)
ax_00.axes.get_yaxis().set_ticks([])
ax_00.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)
ax_01.set_ylim(0, 1)
ax_01.tick_params(labelsize=fontsize)
ax_01.set_xlim(-x_lim, x_lim)
ax_01.axes.get_yaxis().set_ticks([])
ax_01.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)

# Adjust observation plots
ax_02.set_ylim(0, 25)
ax_02.tick_params(labelsize=fontsize)
ax_02.set_xlim(-x_lim, x_lim)
ax_02.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)
ax_02.set_ylabel(r'$p(o_t)$', fontsize=fontsize)
ax_03.set_ylim(0, 25)
ax_03.tick_params(labelsize=fontsize)
ax_03.set_xlim(-x_lim, x_lim)
ax_03.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)
ax_03.set_ylabel(r'$p(o_t)$', fontsize=fontsize)

# Adjust properties of the plots
ax_04 = bs_plot(ax_04, cval_ind, hs)
ax_05 = bs_plot(ax_05, cval_ind, ms)

ax_04.set_ylim(-0.1, 1.1)
ax_04.tick_params(labelsize=fontsize)
ax_04.set_xlim(-x_lim, x_lim)
ax_04.set_ylabel(r'$\pi_1$', fontsize=fontsize)
ax_04.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)
ax_05.set_ylim(-0.1, 1.1)
ax_05.tick_params(labelsize=fontsize)
ax_05.set_xlim(-x_lim, x_lim)
ax_05.set_ylabel(r'$\pi_1$', fontsize=fontsize)
ax_05.set_xlabel(r'Contrast Difference ($d_t$)', fontsize=fontsize)


# B) Demonstration of a single block
# ----------------------------------

T = 25
B = 1
sigma = ms
agent = 1
beta = 100

# Simulate data
df = gb_simulation(T, B, sigma, agent, beta)

# x-axis
x = np.linspace(1, T, T)

# State
ax_10.plot(x, df['s_t'], 'o', color='k', markersize=markersize)
ax_10.set_ylabel('State', fontsize=fontsize)
ax_10.set_ylim([-0.2, 1.2])
ax_10.tick_params(labelsize=fontsize)
ax_10.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Presented contrast difference
ax_11.bar(x, df['u_t'], color='k', width=0.2)
ax_11.set_ylabel('Contrast\n Difference', fontsize=fontsize)
ax_11.set_ylim([-0.1, 0.1])
ax_11.tick_params(labelsize=fontsize)
ax_11.axhline(0, color='black', lw=0.5, linestyle='--')
ax_11.set_xlabel('Trial', fontsize=fontsize)

# Belief state
ax_12.bar(x, df['pi_1'], color='k', width=0.2)
ax_12.set_ylim([0, 1])
ax_12.set_ylabel('Belief\nState', fontsize=fontsize)
ax_12.tick_params(labelsize=fontsize)
ax_12.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Perceptual decision
ax_13.plot(x, df['d_t'], 'o', color='k', markersize=markersize)
ax_13.set_ylabel('Perceptual\nDecision', fontsize=fontsize)
ax_13.set_ylim([-0.2, 1.2])
ax_13.tick_params(labelsize=fontsize)
ax_13.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Conditional Expected value
l1, = ax_14.plot(x, df['v_a_0'], markersize=markersize, color='k', linestyle='-', label='HHZ 1')
l2, = ax_14.plot(x, df['v_a_1'], markersize=markersize, color='k', linestyle='--')
ax_14.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_14.set_ylabel('Conditional\nExpected\n Value', fontsize=fontsize)
ax_14.legend([l1, l2], [r'$a_t = 0$', '$a_t = 1$'], loc=3, frameon=True)
ax_14.set_ylim([0, 1])
ax_14.tick_params(labelsize=fontsize)
ax_14.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Economic decision
ax_15.plot(x[df['a_t'] == 0], df['a_t'][df['a_t'] == 0], 'o', color='k', markersize=markersize)
ax_15.plot(x[df['a_t'] == 1], df['a_t'][df['a_t'] == 1], 'o', color='k', markersize=markersize)
ax_15.set_ylabel('Economic\nDecision', fontsize=fontsize)
ax_15.set_ylim([-0.2, 1.2])
ax_15.tick_params(labelsize=fontsize)
ax_15.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Reward
ax_16.plot(x, df['r_t'], 'ko', markersize=markersize)
ax_16.set_ylim([-0.2, 1.2])
ax_16.tick_params(labelsize=fontsize)
ax_16.set_ylabel('Reward', fontsize=fontsize)
ax_16.set_xlabel(r'Trial ($t$)', fontsize=fontsize)

# Expected value
ax_17.plot(x, df['e_mu_t'], markersize=markersize, color='k')
ax_17.set_ylabel('Expected\n Value', fontsize=fontsize)
ax_17.set_ylim([0.4, 1])
ax_17.tick_params(labelsize=fontsize)
ax_17.axhline(0.8, color='black', lw=0.5, linestyle='--')
ax_17.set_xlabel(r'Trial ($t$)', fontsize=fontsize)


# C) Comparison of performance of A1 and A2
# and the different evolution of expected values
# ----------------------------------------------

T = 25
B = 1000
sigma = ms
beta = 5  # beta parameter of softmax choice rule

# Simulate data with integral model
agent = 1
df_subj_int = gb_simulation(T, B, sigma, agent, beta)

# Simulate data with categorical model
agent = 2
df_subj_cat = gb_simulation(T, B, sigma, agent, beta)

# Compute mean performance of both models
A1_mean = df_subj_int.groupby(df_subj_int['t'])['corr'].mean()
A2_mean = df_subj_cat.groupby(df_subj_int['t'])['corr'].mean()

# Plot mean performance
ax_06.plot(x, A1_mean, color='k')
ax_06.plot(x, A2_mean,  color='k', linestyle='--')
ax_06.tick_params(labelsize=fontsize)
ax_06.set_ylim([0.5, 1])
ax_06.tick_params(labelsize=fontsize)
ax_06.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
ax_06.set_ylabel('Economic Choice Performance', fontsize=fontsize)
ax_06.legend(["Agent A1", "Agent A2"], loc=1, fontsize=fontsize)

# Comparison of evolution of expected value of A1 and A2
# ------------------------------------------------------

B = 10  # number of blocks

# Simulate data with A1
agent = 1
df_subj_a1 = gb_simulation(T, B, sigma, agent, beta)
df_subj_a1['t'] = df_subj_a1['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean expected values
mean_corr_a1 = df_subj_a1.groupby(df_subj_int['t'])['e_mu_t'].mean()
# Plot all expected values
ax_07.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_07.axhline(0.8, color='black', lw=0.5, linestyle='--')
for _, group in df_subj_a1.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_07, legend=False, color='k', linewidth=1)

ax_07.set_ylim([0.2, 1])
ax_07.tick_params(labelsize=fontsize)
ax_07.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
ax_07.set_ylabel(r'Expected Value ($E_{\mu}$)', fontsize=fontsize)
ax_07.plot(x, mean_corr_a1, linewidth=4, color=scalar_map.to_rgba(cval[-1]))

# Simulate data with categorical model
agent = 2
df_subj_a2 = gb_simulation(T, B, sigma, agent, beta)
df_subj_a2['t'] = df_subj_a2['t']+1  # add 1 to trials to start plot with 1 instead of 0

# Compute mean expected values
mean_corr_a2 = df_subj_a2.groupby(df_subj_a2['t'])['e_mu_t'].mean()

# Plot all expected values
ax_18.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax_18.axhline(0.8, color='black', lw=0.5, linestyle='--')
for _, group in df_subj_a2.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax_18, legend=False, color='k', linewidth=1)
ax_18.set_ylim([0.2, 1])
ax_18.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
ax_18.set_ylabel(r'Expected Value ($E_{\mu}$)', fontsize=fontsize)
ax_18.plot(x, mean_corr_a2, linewidth=4, color=scalar_map.to_rgba(cval[-1]))

# Use figure space more efficiently
plt.tight_layout()
sns.despine()

# Save plot
savename = 'gb_figures/Fig3.pdf'
plt.savefig(savename)

# Demonstrate evolution of expected value for belief state model
# --------------------------------------------------------------

T = 100  # number of trials
B = 5  # number of blocks
sigma = ls  # 0.06 ** 2  # sensory noise
agent = 4  # Belief state model
beta = 100

# Simulate data
df_subj_bs = gb_simulation(T, B, sigma, agent, beta)

# Extract expected values
mean_corr_bs = df_subj_bs.groupby(df_subj_bs['t'])['e_mu_t'].mean()

# Plot all expected values
# ------------------------
plt.figure()
ax = plt.gca()
x = np.linspace(1, T, T)
plt.axhline(0.5, color='black', lw=0.5, linestyle='--')
plt.axhline(0.8, color='black', lw=0.5, linestyle='--')
for _, group in df_subj_bs.groupby('block'):
    group.plot(x='t', y='e_mu_t', ax=ax, legend=False, color='k', linewidth=1)
    plt.ylim([0, 1])
    plt.xlabel('Trial')
    plt.ylabel('Expected Value')
plt.plot(x, mean_corr_bs, linewidth=4, color=scalar_map.to_rgba(cval[-1]))
black_line = plt.Line2D([], [], color='black', markersize=15, label='Single Simulations')
orange_line = plt.Line2D([], [], color=scalar_map.to_rgba(cval[-1]), markersize=15, label='Mean')
plt.legend(handles=[black_line, orange_line],  loc='lower left')

# Use figure space more efficiently
plt.tight_layout()

# Save plot
savename = 'gb_figures/A0_demo.png'
plt.savefig(savename)

# 4. Comparison of performance of belief state (basic version without integral)
#  and categorical model for three levels of sensory noise and deterministic choice behavior
# ------------------------------------------------------------------------------------------

# Initialize figure
fig = plt.figure(figsize=(9, 4))
ax_1 = plt.subplot2grid((1, 3), (0, 0))
ax_2 = plt.subplot2grid((1, 3), (0, 1))
ax_3 = plt.subplot2grid((1, 3), (0, 2))

T = 25  # number of trials
B = 1000  # number of blocks
sigma = hs
x = np.linspace(1, T, T)

# Simulate data with low sensory noise
agent = 4
beta = 100
df_subj_bs = gb_simulation(T, B, sigma, agent, beta)
mean_corr_bs = df_subj_bs.groupby(df_subj_bs['t'])['corr'].mean()
agent = 2
df_subj_cat = gb_simulation(T, B, sigma, agent, beta)
mean_corr_cat = df_subj_cat.groupby(df_subj_cat['t'])['corr'].mean()

# Compare performance of both models
ax_1.plot(x, mean_corr_bs, linestyle='-', color='k')
ax_1.plot(x, mean_corr_cat, linestyle='--', color='k')
ax_1.set_ylim([0.5, 1])
ax_1.set_xlabel(r'Trial ($t$)')
ax_1.set_ylabel('Economic Choice Performance')
ax_1.legend(["Agent A1", "Agent A2"], loc=0)
ax_1.set_title(r'$\sigma = %s$' % sigma)

# Simulate data with medium sensory noise
sigma = ms
agent = 4
df_subj_bs = gb_simulation(T, B, sigma, agent, beta)
mean_corr_bs = df_subj_bs.groupby(df_subj_bs['t'])['corr'].mean()
agent = 2
df_subj_cat = gb_simulation(T, B, sigma, agent, beta)
mean_corr_cat = df_subj_cat.groupby(df_subj_cat['t'])['corr'].mean()

# Compare performance
ax_2.plot(x, mean_corr_bs, linestyle='-', color='k')
ax_2.plot(x, mean_corr_cat, linestyle='--', color='k')
ax_2.set_ylim([0.5, 1])
ax_2.set_xlabel(r'Trial ($t$)')
ax_2.set_ylabel('Economic Choice Performance')
ax_2.set_title(r'$\sigma = %s$' % sigma)

# Simulate data with high sensory noise
sigma = ls
agent = 4
df_subj_bs = gb_simulation(T, B, sigma, agent, beta)
mean_corr_bs = df_subj_bs.groupby(df_subj_bs['t'])['corr'].mean()
agent = 2
df_subj_cat = gb_simulation(T, B, sigma, agent, beta)
mean_corr_cat = df_subj_cat.groupby(df_subj_cat['t'])['corr'].mean()

# Compare performance
ax_3.plot(x, mean_corr_bs, linestyle='-', color='k')
ax_3.plot(x, mean_corr_cat, linestyle='--', color='k')
ax_3.set_ylim([0.5, 1])
ax_3.set_xlabel(r'Trial ($t$)')
ax_3.set_ylabel('Economic Choice Performance')
ax_3.set_title(r'$\sigma = %s$' % sigma)

# Adjust figure space
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.10, right=0.9, hspace=0.25, wspace=0.35)
sns.despine()

# Add figure labels
fig.text(0.02, 0.82, "A)", horizontalalignment='left', verticalalignment='center')
fig.text(0.33, 0.82, "B)", horizontalalignment='left', verticalalignment='center')
fig.text(0.62, 0.82, "C)", horizontalalignment='left', verticalalignment='center')


savename = 'gb_figures/SM_Fig7.pdf'
plt.savefig(savename)

# Show all plots
plt.show()
