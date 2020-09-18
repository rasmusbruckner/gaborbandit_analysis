# This script implements the sampling-based validation of the analytical results of agent A4

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from GbTaskVars import TaskVars
from GbTask import Task
from GbAgentVars import AgentVars
from GbAgent import Agent
from PHat import PHat
from gb_cumcoeff import gb_cumcoeff
from gb_invcumcoeff import gb_invcumcoeff
from gb_plot_utils import label_subplots, cm2inch
from time import sleep
from tqdm import tqdm
import sys

# Use Latex for matplotlib
pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": [],
    "axes.labelsize": 6,
    "font.size": 6,
    "legend.fontsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.titlesize": 6,
    "pgf.rcfonts": False,
    "figure.dpi": 100,
    "text.latex.unicode": True,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ]
}

# Update parameters
matplotlib.rcParams.update(pgf_with_latex)

# Set random number generator for reproducible results
np.random.seed(123)

# Todo: in the first part the validation we not yet sample. Add this in next version.

# Simulation parameters
T = 26  # Number of trials
n_samples = int(1e6)  # Number of samples
n_bins = int(1e2)  # Number of bins for density approximation

# Model parameters
kappa = 0.08  # Maximal contrast difference value
sigma = 0.04  # Perceptual sensitivity

# Initialize task and agent objects
# ---------------------------------

# Task parameters in TaskVars object
task_vars = TaskVars()
task_vars.T = T
task_vars.B = 1
task = Task(task_vars)

# Agent parameters in AgentVars object
agent_vars = AgentVars()
agent_vars.agent = 1
agent_vars.sigma = sigma
agent_vars.task_agent_analysis = False
agent = Agent(agent_vars)
agent.d_t = 0  # Fix perceptual decision to d_t = 0
agent.a_t = 0  # Fix action to a_t = 0

# Sampling-based approximation object
p_hat = PHat(n_bins)

# Number of variables x number of samples x number of time points sample matrix
S = np.full([5, n_samples, T], np.nan)

# Simulation scenarios / observed random variable values
# ------------------------------------------------------

O_list = list()
R_list = list()

# Reliable co-occurrence of a clear positive contrast difference and no reward
O_list.append(np.repeat([0.08], T))
R_list.append(np.repeat([0], T))

# Reliable co-occurrence of a clear positive contrast difference and reward
O_list.append(np.repeat([0.08], T))
R_list.append(np.repeat([1], T))

# Clear negative contrast difference and alternating reward
O_list.append(np.repeat([-0.08], T))
R_list.append(np.tile([1, 0], [np.int(T/2)]))

# Weak positive contrast difference and alternating reward
O_list.append(np.repeat([0.02], T))
R_list.append(np.tile([1, 0], [np.int(T/2)]))

# Alternating weak positive and negative contrast differences and alternating reward
O_list.append(np.repeat([0.02, -0.02], [np.int(T/2)]))
R_list.append(np.tile([1, 0], [np.int(T/2)]))

# Resulting number of simulation scenarios
n_sim = len(O_list)

# Initialize variables for the computation of the estimation bias
mu_bias_mean = np.full(T * n_sim, np.nan)
bs0_bias_mean = np.full(T * n_sim, np.nan)
bs1_bias_mean = np.full(T * n_sim, np.nan)
rt0_bias_mean = np.full(T * n_sim, np.nan)
rt1_bias_mean = np.full(T * n_sim, np.nan)

# Initialize counter as index across trials and simulations
counter = 0

# Cycle over simulations
for sim in range(0, n_sim):

    # Initialize progress bar
    sleep(0.1)
    print('\nRunning simulation %d:' % sim)
    sleep(0.1)
    pbar = tqdm(total=T)

    # Initialize polynomial coefficients
    C_t = []

    # Initialize integral of p(\mu)
    I_t = []  # np.zeros(1)

    # Extract observations and rewards
    O_t = O_list[sim]
    R_t = R_list[sim]

    # Evaluate coefficients
    C_t.append(np.array([1]))
    poly_ind_int = np.polyint(C_t[0])
    poly_eval = np.polyval(poly_ind_int, [0, 1])
    I_t.append(np.diff(poly_eval))  # Integral of p_0

    # Evaluate observed RV conditional prior distributions p_1, ..., p_{T-1}
    for t in range(1, T+1):

        # Set current agent variables
        agent.c_t = np.asarray(C_t[t-1])  # coefficient
        agent.o_t = O_t[t-1]  # observation  todo: here we should sample

        # Update polynomial coefficients
        agent.learn(R_t[t-1])
        C_t.append(agent.c_t)

        # Compute integral over \mu
        poly_ind_int = np.polyint(C_t[t])
        poly_eval = np.polyval(poly_ind_int, [0, 1])
        I_t.append(np.diff(poly_eval))

    # Density check
    if not any((x == 1).all() for x in I_t):
        sys.exit("Density check failed")

    # Sample RVs
    # ----------

    # Cycle over trials
    for t in range(0, T):

        # Cycle over samples
        for s in range(0, n_samples):

            # Compute cumulative density function of p_t(\mu)
            c_F = gb_cumcoeff(C_t[t])

            # p_{t-1}(\mu)
            y = np.random.uniform(0, 1)
            S[0, s, t] = gb_invcumcoeff(y, c_F)

            # p(s_t) := B(s_t;0.5)
            task.state_sample()
            S[1, s, t] = task.s_t

            # p(c_t|s_t = 0)
            task.contrast_sample()
            S[2, s, t] = task.c_t

            # p(o_t|c_t)
            agent.observation_sample(task.c_t)   # todo: here we do sample
            S[3, s, t] = agent.o_t

            # p(r_t|s_t)
            if task.s_t == 0:
                S[4, s, t] = np.random.binomial(1, S[0, s, t])
            else:
                S[4, s, t] = np.random.binomial(1, 1 - S[0, s, t])

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Visualization for each trial
    for t in range(0, T):

        # Estimate p_{t-1}(o_t|s_t) and p_{t-1}(\mu|r_t,o_t)
        p_hat.S = S[:, :, t]
        p_hat.gb_p_hat()

        # Extract samples
        S[:, :, t] = p_hat.S

        # X-axis according to number of bins
        x = np.arange(n_bins)  # np.linspace(0, 99, 100)

        # Evaluate p_{t-1}(\mu_t|r_t, o_t) analytically
        pdf_mu_giv_r_t_o_t = np.full([len(p_hat.mu), len(x), 2], np.nan)

        for r in range(0, len(p_hat.r_e)):

            for i in range(0, len(p_hat.o)):

                # Set current agent variables
                agent.c_t = np.asarray(C_t[t])  # coefficient
                agent.o_t = p_hat.o[i]  # observation todo: adjust for sampling-based validation

                # p_{t-1}(\mu|r_t = 0, o_t) analytical result
                agent.learn(p_hat.r_e[r])
                pdf_mu_giv_r_t_o_t[:, i, r] = np.polyval(agent.c_t, p_hat.mu)

        # Initialize figure
        # -----------------
        fig_width = 15
        fig_height = 20
        f = plt.figure(figsize=cm2inch(fig_width, fig_height))

        ax_0 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
        ax_1 = plt.subplot2grid((4, 4), (0, 2))
        ax_2 = plt.subplot2grid((4, 4), (0, 3))
        ax_3 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
        ax_4 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
        ax_5 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
        ax_6 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
        ax_7 = plt.subplot2grid((4, 4), (3, 0))
        ax_8 = plt.subplot2grid((4, 4), (3, 1))
        ax_9 = plt.subplot2grid((4, 4), (3, 2))
        ax_10 = plt.subplot2grid((4, 4), (3, 3))
        barcolor = "slategray"

        # p_t(\mu)
        # --------

        # Analytical solution
        p_mu = np.polyval(C_t[t], p_hat.mu)

        # Compute and plot sampling solution
        n = ax_0.hist(S[0, :, t], bins=n_bins, color=barcolor, density=True)
        n_array = np.asarray(n[0])

        # Compute estimation bias
        mu_bias = p_mu - n_array
        mu_bias_mean[counter] = np.mean(mu_bias)

        # Plot analytical solution
        ax_0.plot(p_hat.mu, p_mu, color='k')
        ax_0.set_xlabel(r'$\mu$')
        ax_0.set_ylabel(r'$p_{%d}(\mu)$' % t)

        # p(s_t|o_t)
        # ----------

        # Analytical solution
        pi_0, pi_1 = agent.p_s_giv_o(p_hat.o)

        # Compute estimation bias
        bs0_bias_mean[counter] = np.mean(p_hat.p_s_giv_o_hat[0, :] - pi_0)
        bs1_bias_mean[counter] = np.mean(p_hat.p_s_giv_o_hat[1, :] - pi_1)

        # Plot sampling solution for p(s_t = 0|o_t)
        ax_1.plot(p_hat.o, p_hat.p_s_giv_o_hat[0, :], color='slategray')
        ax_1.fill_between(p_hat.o, 0, p_hat.p_s_giv_o_hat[0, :], facecolor=barcolor)
        ax_1.set_xlabel(r'$o_{%d}$' % (t+1))
        ax_1.set_ylabel(r'$p_{%d}(s_%d=0|o_{%d})$' % (t, t+1, t+1))

        # Plot sampling solution for p(s_t = 1|o_t)
        ax_2.plot(p_hat.o, p_hat.p_s_giv_o_hat[1, :], color='slategray')
        ax_2.fill_between(p_hat.o, 0, p_hat.p_s_giv_o_hat[1, :], facecolor=barcolor)
        ax_2.set_xlabel(r'$o_{%d}$' % (t + 1))
        ax_2.set_ylabel(r'$p_{%d}(s_%d=1|o_{%d})$' % (t, t + 1, t + 1))

        # Plot analytical solution for p(s_t|o_t)
        ax_1.plot(p_hat.o, pi_0, color='k')
        ax_2.plot(p_hat.o, pi_1, color='k')

        # p_{t-1}(\mu|r_t = 0, o_t) and p_{t-1}(\mu|r_t = 1, o_t)
        # -----------------------------------------------------------

        # Compute estimation bias
        rt0_bias_mean[counter] = np.mean(p_hat.p_mu_giv_r_o_hat[:, :, 0] - pdf_mu_giv_r_t_o_t[:, :, 0])
        rt1_bias_mean[counter] = np.mean(p_hat.p_mu_giv_r_o_hat[:, :, 1] - pdf_mu_giv_r_t_o_t[:, :, 1])

        # Plot sampling and analytical solution
        ax_3.set_title(r'Sampling: $p_{%d}(\mu|r_{%d}=0, o_{%d})$' % (t, t + 1, t + 1))
        ax_4.set_title(r'Sampling: $p_{%d}(\mu|r_{%d}=1, o_{%d})$' % (t, t + 1, t + 1))
        ax_5.set_title(r'Analytical: $p_{%d}(\mu|r_{%d}=0, o_{%d})$' % (t, t + 1, t + 1))
        ax_6.set_title(r'Analytical: $p_{%d}(\mu|r_{%d}=1, o_{%d})$' % (t, t + 1, t + 1))
        im_1 = ax_3.imshow(p_hat.p_mu_giv_r_o_hat[:, :, 0], extent=[0, 100, 0, 1],
                           aspect='auto', origin='lower', cmap="bone")
        im_2 = ax_4.imshow(p_hat.p_mu_giv_r_o_hat[:, :, 1], extent=[0, 100, 0, 1],
                           aspect='auto', origin='lower', cmap="bone")
        im_3 = ax_5.imshow(pdf_mu_giv_r_t_o_t[:, :, 0], extent=[0, 100, 0, 1],
                           aspect='auto', origin='lower', cmap="bone")
        im_4 = ax_6.imshow(pdf_mu_giv_r_t_o_t[:, :, 1], extent=[0, 100, 0, 1],
                           aspect='auto', origin='lower', cmap="bone")
        a = ax_3.get_xticks().tolist()
        a[0], a[1], a[2], a[3], a[4], a[5] = '-0.1', '-0.06', '-0.02', '0.02', '0.06', '0.1'
        ax_3.set_xticklabels(a)
        ax_4.set_xticklabels(a)
        ax_5.set_xticklabels(a)
        ax_6.set_xticklabels(a)
        ax_3.set_xlabel(r'$o_{%d}$' % (t + 1))
        ax_4.set_xlabel(r'$o_{%d}$' % (t + 1))
        ax_5.set_xlabel(r'$o_{%d}$' % (t + 1))
        ax_6.set_xlabel(r'$o_{%d}$' % (t + 1))

        # Plot approximation and analytical result for four selected o_t^*
        # ----------------------------------------------------------------

        # Observation of interest indices
        ooi_idx = np.array([0.1 * p_hat.o_nb, 0.9 * p_hat.o_nb, 0.5 * p_hat.o_nb, 0.48 * p_hat.o_nb]).astype(int)

        # Plot these scenarios
        ax_7.plot(p_hat.mu, pdf_mu_giv_r_t_o_t[:, ooi_idx[0], 0], color='k')
        ax_7.bar(p_hat.mu, p_hat.p_mu_giv_r_o_hat[:, ooi_idx[0], 0], width=0.01, color=barcolor)
        ax_7.set_title(r'$p_{%d}(\mu|r_{%d}=1, o_{%d}=-0.08)$' % (t, t + 1, t + 1))
        ax_7.set_xlabel(r'$\mu$')
        ax_7.set_ylim(0, 19)
        ax_8.plot(p_hat.mu, pdf_mu_giv_r_t_o_t[:, ooi_idx[1], 0], color='k')
        ax_8.bar(p_hat.mu, p_hat.p_mu_giv_r_o_hat[:, ooi_idx[1], 0], width=0.01, color=barcolor)
        ax_8.set_title(r'$p_{%d}(\mu|r_{%d}=1, o_{%d}=0.08)$' % (t, t + 1, t + 1))
        ax_8.set_xlabel(r'$\mu$')
        ax_8.set_ylim(0, 19)
        ax_9.plot(p_hat.mu, pdf_mu_giv_r_t_o_t[:, ooi_idx[2], 0], color='k')
        ax_9.bar(p_hat.mu, p_hat.p_mu_giv_r_o_hat[:, ooi_idx[2], 0], width=0.01, color=barcolor)
        ax_9.set_title(r'$p_{%d}(\mu|r_{%d}=1, o_{%d}=-0.001)$' % (t, t + 1, t + 1))
        ax_9.set_xlabel(r'$\mu$')
        ax_9.set_ylim(0, 19)
        ax_10.plot(p_hat.mu, pdf_mu_giv_r_t_o_t[:, ooi_idx[3], 0], color='k')
        ax_10.bar(p_hat.mu, p_hat.p_mu_giv_r_o_hat[:, ooi_idx[3], 0], width=0.01, color=barcolor)
        ax_10.set_title(r'$p_{%d}(\mu|r_{%d}=1, o_{%d}=-0.005)$' % (t, t + 1, t + 1))
        ax_10.set_xlabel(r'$\mu$')
        ax_10.set_ylim(0, 19)

        # Use figure space more efficiently
        sns.despine()

        # Adjust figure space
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=1, wspace=1)

        # Add figure labels
        # -----------------

        # Label letters
        texts = ['a', 'b ', 'c ', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

        # Add labels
        label_subplots(f, texts)

        # Save plot
        savename = 'gb_figures/sampling_validation_final/sim_%s_t_%s.pdf' % (sim, t)
        plt.savefig(savename, dpi=400, transparent=True)
        plt.close()

        # Update counter
        counter += 1

# Generate plot that shows estimation bias across all simulations
# ---------------------------------------------------------------

# Concatenate all estimation biases of interest
data = np.concatenate((mu_bias_mean, bs0_bias_mean, bs1_bias_mean, rt0_bias_mean, rt1_bias_mean), axis=0)
df = pd.DataFrame(data=data, columns=["param"])
df['type'] = np.repeat(np.arange(5), T*n_sim)

# Plot estimation biases
plt.figure()
ax = sns.barplot(x='type', y='param', data=df, ci='sd', color='k')
plt.ylabel("Estimation Bias")
plt.ylim(-5, 5)
plt.xlabel('Variable')
a = ax.get_xticks().tolist()
a[0], a[1], a[2], a[3], a[4] = r'$p_t(\mu)$', r'$p(s_t = 0|o_t)$', r'$p(s_t = 1|o_t)$',\
                               r'$p_{t-1}(\mu|r_t = 0, o_t)$', r'$p_{t-1}(\mu|r_t = 1, o_t)$'
ax.set_xticklabels(a, rotation=30)
plt.tight_layout()
sns.despine()

savename = 'gb_figures/SM_gb_figure_2.pdf'
# plt.savefig(savename)
plt.close()
