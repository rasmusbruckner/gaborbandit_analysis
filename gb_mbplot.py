""" This script plots the model-based analyses and the results of the recovery studies

    1. Figure 4
    2. SM Figure 5
    3. SM Figure 6
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from scipy import stats
import pickle
import csv
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

# Load participant data
# ---------------------

# Parameter estimates
part_params = pd.read_pickle('gb_data/modelbased.pkl')

# Get all IDs
all_id = list(set(part_params['id']))

# Number of participants and blocks
N = len(all_id)

# Get BIC for each agent model
BIC_A0 = sum(part_params['A0_d_BIC']) + sum(part_params['A0_a_BIC'])
BIC_A1 = sum(part_params['A1_d_BIC']) + sum(part_params['A1_a_BIC'])
BIC_A2 = sum(part_params['A2_d_BIC']) + sum(part_params['A2_a_BIC'])
BIC_A3 = sum(part_params['A3_d_BIC']) + sum(part_params['A3_a_BIC'])

# Participant exceedance probabilities
# ------------------------------------
expr_part = list()
with open("gb_data/exceedance_probs_part.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        expr_part.append(row)
expr_part = [float(i) for i in expr_part[0]]

# Trial-wise average economic choice performance
trial_wise_part = pd.read_pickle('gb_data/trial_wise_part.pkl')

# Model performance
f = open('gb_data/postpred.pkl', 'rb')
post_pred = pickle.load(f)
f.close()

mean_corr_A0  = post_pred[0, :]
mean_corr_A1 = post_pred[1, :]
mean_corr_A2 = post_pred[2, :]
mean_corr_A3 = post_pred[3, :]

# Model recovery BIC's
df_bic0 = pd.read_pickle('gb_data/df_bic_0.pkl')
df_bic1 = pd.read_pickle('gb_data/df_bic_1.pkl')
df_bic2 = pd.read_pickle('gb_data/df_bic_2.pkl')
df_bic3 = pd.read_pickle('gb_data/df_bic_3.pkl')

# Get BIC's for A0
bic0_d = df_bic0.groupby(['agent'])['d_BIC'].sum()
bic0_a = df_bic0.groupby(['agent'])['a_BIC'].sum()
A0_0 = bic0_d[0] + bic0_a[0]
A0_1 = bic0_d[1] + bic0_a[1]
A0_2 = bic0_d[2] + bic0_a[2]
A0_3 = bic0_d[3] + bic0_a[3]

# Get BIC's for A1
bic_d = df_bic1.groupby(['agent'])['d_BIC'].sum()
bic_a = df_bic1.groupby(['agent'])['a_BIC'].sum()
A1_0 = bic_d[0] + bic_a[0]
A1_1 = bic_d[1] + bic_a[1]
A1_2 = bic_d[2] + bic_a[2]
A1_3 = bic_d[3] + bic_a[3]

# Get BIC's for A2
bic_d = df_bic2.groupby(['agent'])['d_BIC'].sum()
bic_a = df_bic2.groupby(['agent'])['a_BIC'].sum()
A2_0 = bic_d[0] + bic_a[0]
A2_1 = bic_d[1] + bic_a[1]
A2_2 = bic_d[2] + bic_a[2]
A2_3 = bic_d[3] + bic_a[3]

# Get BIC's for A2
bic_d = df_bic3.groupby(['agent'])['d_BIC'].sum()
bic_a = df_bic3.groupby(['agent'])['a_BIC'].sum()
A3_0 = bic_d[0] + bic_a[0]
A3_1 = bic_d[1] + bic_a[1]
A3_2 = bic_d[2] + bic_a[2]
A3_3 = bic_d[3] + bic_a[3]

# Model recovery exceedance probabilities
# ---------------------------------------
expr_recov = list()
with open("gb_data/exceedance_probs_recov.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        expr_recov.append(row)
expr_A0 = [float(i) for i in expr_recov[0]]
expr_A1 = [float(i) for i in expr_recov[1]]
expr_A2 = [float(i) for i in expr_recov[2]]
expr_A3 = [float(i) for i in expr_recov[3]]

# Parameter recovery
param_recov = pd.read_pickle('gb_data/param_recov.pkl')

# Figure 4
# --------

# Define colormap
new_cmap = truncate_colormap(plt.get_cmap('bone'), 0.2, 0.7)

# Create range of colors for range of contrast differences
cval_ind = np.arange(100)
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=new_cmap)

blue2 = scalar_map.to_rgba(cval_ind[20])  # color for Experiment 1
blue1 = scalar_map.to_rgba(cval_ind[60])  # color for Experiment 2
blue3 = scalar_map.to_rgba(cval_ind[80])  # color for Experiment 3

# Prepare figure
fig4 = plt.figure(figsize=(10, 6))
ax_0 = plt.subplot2grid((2, 6), (0, 0))
ax_1 = plt.subplot2grid((2, 6), (0, 1))
ax_2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
ax_3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
ax_4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax_5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

# Compute parameter statistics
sigma_mean = part_params['sigma'].mean()
sigma_std = part_params['sigma'].std()
sigma_sem = sigma_std/np.sqrt(N)
beta_mean = part_params['beta'].mean()
beta_std = part_params['beta'].std()
beta_sem = beta_std/np.sqrt(N)
lambda_mean = part_params['lambda'].mean()
lambda_std = part_params['lambda'].std()
lambda_sem = lambda_std/np.sqrt(N)

# Model comparison
# ----------------
# Plot BIC's
ax_0.bar([1, 2, 3, 4], [BIC_A0, BIC_A1, BIC_A2, BIC_A3], facecolor=blue1, edgecolor='k', alpha=1)
ax_0.set_ylim(-22000, -11000)
ax_0.set_ylabel(r'$\sum$ BIC')
ax_0.set_xticks([1, 2, 3, 4])
a = ax_0.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_0.set_xticklabels(a)

# Plot exceedance probabilities
ax_1.bar([1, 2, 3, 4], expr_part, facecolor=blue1, alpha=1, edgecolor='k')
ax_1.set_ylim(0, 1)
ax_1.set_ylabel('Exceedance Probability')
ax_1.set_xticks([1, 2, 3, 4])
a = ax_1.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_1.set_xticklabels(a)

# Plot participant parameters
# ---------------------------
# Sigma
ax_3.bar(np.linspace(0, N-1, N), part_params['sigma_exp3'], facecolor=blue1, alpha=1, edgecolor='k')
ax_3.set_xlabel('Participant')
ax_3.set_ylabel(r'$\sigma$', rotation=0, labelpad=10)
ax_3.axhline(part_params['sigma_exp3'].mean(), color='k', lw=1.5, linestyle='--')

# Beta
ax_4.bar(np.linspace(0, N-1, N), part_params['beta'], facecolor=blue1, alpha=1, edgecolor='k')
ax_4.set_xlabel('Participant')
ax_4.set_ylabel(r'$\beta$', rotation=0, labelpad=10)
ax_4.axhline(part_params['beta'].mean(), color='k', lw=1.5, linestyle='--')

# Lambda
ax_5.bar(np.linspace(0, N-1, N), part_params['lambda'], facecolor=blue1, alpha=1, edgecolor='k')
ax_5.set_xlabel('Participant')
ax_5.set_ylabel(r'$\lambda$', rotation=0, labelpad=10)
ax_5.axhline(part_params['lambda'].mean(), color='k', lw=1.5, linestyle='--')

# Plot posterior predictive checks
x = np.linspace(1, 25, 25)
ax_2.plot(x, trial_wise_part['trial_mean'], color='black', zorder=1, linewidth=3)
ax_2.fill_between(x, trial_wise_part['trial_mean']-trial_wise_part['trial_sd'],
                  trial_wise_part['trial_mean']+trial_wise_part['trial_sd'], facecolor=blue1, alpha=0.5, edgecolor='k')
ax_2.plot(x, np.mean(mean_corr_A0, 0), color='k', linestyle='--', linewidth=2)
a1 = ax_2.plot(x, np.mean(mean_corr_A1, 0), color='b', linestyle='-', linewidth=2)
a2 = ax_2.plot(x, np.mean(mean_corr_A2, 0), color='y', linestyle='-', linewidth=2)
color_seq = np.tile([0, 1], [11])
points = np.array([x, np.mean(mean_corr_A3, 0)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
cmap = ListedColormap(['b', 'y'])  # use combination of blue and yellow
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_zorder(100)  # make sure line for A3 is in the foreground
lc.set_array(color_seq)
lc.set_linewidth(3)
line = ax_2.add_collection(lc)
ax_2.set_xlabel('Trial')
ax_2.set_ylabel('Performance')
ax_2.set_ylim([0.3, 1])
ax_2.set_xlabel('Trial')
ax_2.set_ylabel('Economic Choice Performance')

# Make custom legend
part = Line2D([0], [0], color='black', lw=2)
a0 = Line2D([0], [0], color='black', lw=2, linestyle='--')
yellow = Line2D([0], [0], color=cmap(0.), lw=2)
blue = Line2D([0], [0], color=cmap(.5), lw=2)
ax_2.legend([part, a0, yellow, blue, (yellow, blue)], ['Participants', 'Agent A0', 'Agent A1', 'Agent A2', 'Agent A3'],
            handler_map={tuple: HandlerTuple(ndivide=None)})

# Adjust figure space
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.10, right=0.9, hspace=0.5, wspace=0.75)
sns.despine()

# Add figure labels
fig4.text(0.02, 0.95, "A)", horizontalalignment='left', verticalalignment='center')
fig4.text(0.2, 0.95, "B)", horizontalalignment='left', verticalalignment='center')
fig4.text(0.34, 0.95, "C)", horizontalalignment='left', verticalalignment='center')
fig4.text(0.02, 0.4, "D)", horizontalalignment='left', verticalalignment='center')
fig4.text(0.34, 0.4, "E)", horizontalalignment='left', verticalalignment='center')
fig4.text(0.63, 0.4, "F)", horizontalalignment='left', verticalalignment='center')

# Save figure
savename = 'gb_figures/Fig4.pdf'
plt.savefig(savename)

# Plot correlation between sigma and lambda
# -----------------------------------------

plt.figure()
# r, p = stats.pearsonr(part_params['sigma'], part_params['lambda'])
r, p = stats.pearsonr(part_params['sigma_exp3'], part_params['lambda'])
sns.regplot(part_params['lambda'], part_params['sigma_exp3'], color=blue1)
plt.title('r = %s, P = %s' % (np.round(r, 3), np.round(p, 3)))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\sigma$')

# SM Figure 5
# -----------

fig = plt.figure(30, figsize=(6.4, 7.2))
ax_00 = fig.add_subplot(421)
ax_01 = fig.add_subplot(422)
ax_10 = fig.add_subplot(423)
ax_11 = fig.add_subplot(424)
ax_20 = fig.add_subplot(425)
ax_21 = fig.add_subplot(426)
ax_30 = fig.add_subplot(427)
ax_31 = fig.add_subplot(428)

# Adjust figure space
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.10, right=0.9, hspace=0.5, wspace=0.35)
sns.despine()

# Agent A0
# --------
# Plot BIC
ax_00.bar([0, 1, 2, 3], [A0_0, A0_1, A0_2, A0_3], facecolor=blue3, edgecolor='k')
ax_00.set_ylabel(r'$\sum$ BIC')
ax_00.set_xlabel('Agent')
a = ax_00.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_00.set_xticklabels(a)
ax_00.set_title('Agent A0')
ax_00.set_ylim(-40000, -15000)

# Plot exceedance probabilites
ax_01.bar([0, 1, 2, 3], expr_A0, facecolor=blue3, edgecolor='k')
ax_01.set_ylabel('Exceedance Pr.')
ax_01.set_xlabel('Agent')
a = ax_01.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_01.set_xticklabels(a)
ax_01.set_title('Agent A0')

# Agent A1
# --------
# Plot BIC
ax_10.bar([0, 1, 2, 3], [A1_0, A1_1, A1_2, A1_3], facecolor=blue3, edgecolor='k')
ax_10.set_ylabel(r'$\sum$ BIC')
ax_10.set_xlabel('Agent')
a = ax_10.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_10.set_xticklabels(a)
ax_10.set_title('Agent A1')
ax_10.set_ylim(-25000, -10000)

# Plot exceedance probabilites
ax_11.bar([0, 1, 2, 3], expr_A1, facecolor=blue3, edgecolor='k')
ax_11.set_ylabel('Exceedance Pr.')
ax_11.set_xlabel('Agent')
a = ax_11.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_11.set_xticklabels(a)
ax_11.set_title('Agent A1')

# Agent A2
# --------
# Plot BIC
ax_20.bar([0, 1, 2, 3], [A2_0, A2_1, A2_2, A2_3], facecolor=blue3, edgecolor='k')
ax_20.set_ylabel(r'$\sum$ BIC')
ax_20.set_xlabel('Agent')
a = ax_20.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_20.set_xticklabels(a)
ax_20.set_title('Agent A2')
ax_20.set_ylim(-25000, -10000)

# Plot exceedance probabilites
ax_21.bar([0, 1, 2, 3], expr_A2, facecolor=blue3, edgecolor='k')
ax_21.set_ylabel('Exceedance Pr.')
ax_20.set_xlabel('Agent')
a = ax_21.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_21.set_xticklabels(a)
ax_21.set_title('Agent A2')

# Agent A3
# --------
# Plot BIC
ax_30.bar([0, 1, 2, 3], [A3_0, A3_1, A3_2, A3_3], facecolor=blue3, edgecolor='k')
ax_30.set_ylabel(r'$\sum$ BIC')
ax_30.set_xlabel('Agent')
a = ax_30.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_30.set_xticklabels(a)
ax_30.set_title('Agent A3')
ax_30.set_ylim(-25000, -10000)

# Plot exceedance probabilites
ax_31.bar([0, 1, 2, 3], expr_A3, facecolor=blue3, edgecolor='k')
ax_31.set_ylabel('Exceedance Pr.')
ax_31.set_xlabel('Agent')
a = ax_31.get_xticks().tolist()
a[0], a[1], a[2], a[3] = 'A0', 'A1', 'A2', 'A3'
ax_31.set_xticklabels(a)
ax_31.set_title('Agent A3')

# Use figure space more efficiently
plt.tight_layout()

# Save plot
savename = 'gb_figures/SM_Fig5.pdf'
plt.savefig(savename)

# SM Figure 6
# -----------
fig = plt.figure(20, figsize=(6.4, 2.4))

# Sigma bias
ax = fig.add_subplot(131)
sns.pointplot(x='which_param', y='sigma_bias', data=param_recov, ci='sd', color='k')
ax.set_ylabel("Estimation Bias")
ax.set_ylim(-0.05, 0.05)
ax.set_xticklabels(['.01', '.026', '.042', '.058', '.074', '.09'], rotation=45)
ax.set_xlabel(r'$\sigma$')

# Beta bias
ax = fig.add_subplot(132)
sns.pointplot(x='which_param', y='beta_bias', data=param_recov, ci='sd', color='k')
ax.set_ylabel("Estimation Bias")
ax.set_ylim(-8, 8)
ax.set_xticklabels(['0', '3.6', '7.2', '10.8', '14.4', '18'], rotation=45)
ax.set_xlabel(r'$\beta$')

# Lambda bias
ax = fig.add_subplot(133)
sns.pointplot(x='which_param', y='lambda_bias', data=param_recov, ci='sd', color='k')
ax.set_ylabel("Estimation Bias")
ax.set_ylim(-0.5, 0.5)
ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], rotation=45)
ax.set_xlabel(r'$\lambda$')

# Use figure space more efficiently
plt.tight_layout()
sns.despine()

savename = 'gb_figures/SM_Fig6.pdf'
plt.savefig(savename)

plt.show()
