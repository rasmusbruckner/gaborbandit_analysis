""" This script plots Figure 3

    1. Load and prepare data
    2. Prepare figure
    3. Plot model recovery
    4. Plot parameter recovery
    5. Add subplot labels and save figure
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib
import seaborn as sns
import csv
from latex_plt import latex_plt
from gb_plot_utils import cm2inch, label_subplots, get_bic
import os
from matplotlib import rc
from brokenaxes import brokenaxes
# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# todo:
# cleaning:
# GbEstimation (paar Punkte bei last check und Namenänderung, ansonsten done)
# gb_recovery (paar Punkte bei last check und Namenänderung, ansonsten done)
# gb_recovsim (paar Punkte bei last check und Namenänderung, ansonsten done)
# gb_parallelest (paar Punkte bei last check und Namenänderung, ansonsten done)
# GbTaskVars (am ende namen konsitent machen, ansonsten done)
# GbSimVars (am ende namen konsitent machen, ansonsten done)
# GbEstVars (am ende namen konsitent machen, ansonsten done)
# gb_eval (noch schauen was eval_simulations==False macht, ansonsten done)

# ------------------------
# 1. Load and prepare data
# ------------------------

# Model recovery BIC's
# These data were generated using the script gb_recovery.py
df_bic0 = pd.read_pickle('gb_data/df_bic_0_final.pkl')
df_bic1 = pd.read_pickle('gb_data/df_bic_1_final.pkl')
df_bic2 = pd.read_pickle('gb_data/df_bic_2_final.pkl')
df_bic3 = pd.read_pickle('gb_data/df_bic_3_final.pkl')
df_bic4 = pd.read_pickle('gb_data/df_bic_4_final.pkl')
df_bic5 = pd.read_pickle('gb_data/df_bic_5_final.pkl')
df_bic6 = pd.read_pickle('gb_data/df_bic_6_final.pkl')
div = 10000  # set divisor to improve visibility in plot
A0_0, A0_1, A0_2, A0_3, A0_4, A0_5, A0_6 = get_bic(df_bic0, div)
A1_0, A1_1, A1_2, A1_3, A1_4, A1_5, A1_6 = get_bic(df_bic1, div)
A2_0, A2_1, A2_2, A2_3, A2_4, A2_5, A2_6 = get_bic(df_bic2, div)
A3_0, A3_1, A3_2, A3_3, A3_4, A3_5, A3_6 = get_bic(df_bic3, div)
A4_0, A4_1, A4_2, A4_3, A4_4, A4_5, A4_6 = get_bic(df_bic4, div)
A5_0, A5_1, A5_2, A5_3, A5_4, A5_5, A5_6 = get_bic(df_bic5, div)
A6_0, A6_1, A6_2, A6_3, A6_4, A6_5, A6_6 = get_bic(df_bic6, div)
A0 = [A0_0, A1_0, A2_0, A3_0, A4_0, A5_0, A6_0]
A1 = [A0_1, A1_1, A2_1, A3_1, A4_1, A5_1, A6_1]
A2 = [A0_2, A1_2, A2_2, A3_2, A4_2, A5_2, A6_2]
A3 = [A0_3, A1_3, A2_3, A3_3, A4_3, A5_3, A6_3]
A4 = [A0_4, A1_4, A2_4, A3_4, A4_4, A5_4, A6_4]
A5 = [A0_5, A1_5, A2_5, A3_5, A4_5, A5_5, A6_5]
A6 = [A0_6, A1_6, A2_6, A3_6, A4_6, A5_6, A6_6]

# Get exceedance probabilities
# Exceedance probabilites were computed in Matlab using SPM12
expr_recov = list()
with open("gb_data/exceedance_probs_recov.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        # print(row)
        expr_recov.append(row)
expr_A0 = [float(i) for i in expr_recov[0]]
expr_A1 = [float(i) for i in expr_recov[1]]
expr_A2 = [float(i) for i in expr_recov[2]]
expr_A3 = [float(i) for i in expr_recov[3]]
expr_A4 = [float(i) for i in expr_recov[4]]
expr_A5 = [float(i) for i in expr_recov[5]]
expr_A6 = [float(i) for i in expr_recov[6]]

# Parameter recovery
# These data were generated using the script gb_recovery.py
param_recov_sigma = pd.read_pickle('gb_data/param_recov_new_sigma.pkl')
param_recov_1 = pd.read_pickle('gb_data/param_recov_new_1.pkl')
param_recov_2 = pd.read_pickle('gb_data/param_recov_new_2.pkl')
param_recov_3 = pd.read_pickle('gb_data/param_recov_new_3.pkl')
param_recov_4 = pd.read_pickle('gb_data/param_recov_new_4.pkl')
param_recov_5 = pd.read_pickle('gb_data/param_recov_new_5.pkl')
param_recov_6 = pd.read_pickle('gb_data/param_recov_new_6.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Figure properties
fig_ratio = 1.4
fig_width = 15
fig_height = 13
low_alpha = 0.3
medium_alpha = 0.6
high_alpha = 1
blue_1 = '#46b3e6'
blue_2 = '#4d80e4'
blue_3 = '#2e279d'
green_1 = '#94ed88'
green_2 = '#52d681'
green_3 = '#00ad7c'

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs0 = gridspec.GridSpec(nrows=10, ncols=1, left=0.1, hspace=3, top=.95, right=0.99, wspace=1, bottom=0.08)

# ----------------------
# 3. Plot model recovery
# ----------------------

# Set width of bars
barWidth = 0.1

# Compute position of BIC bars and exceedance probabilities
r1 = np.arange(len(A1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

# Plot BIC bars
gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0:4, 0], hspace=0.7)
ax_00 = plt.Subplot(f, gs01[:2, 0])
f.add_subplot(ax_00)
ax_00.bar(r1, A0, color='k', width=barWidth, edgecolor='white', label='A0')
ax_00.bar(r2, A1, color=blue_1, width=barWidth, edgecolor='white', label='A1')
ax_00.bar(r3, A2, color=blue_2, width=barWidth, edgecolor='white', label='A2')
ax_00.bar(r4, A3, color=blue_3, width=barWidth, edgecolor='white', label='A3')
ax_00.bar(r5, A4, color=green_1, width=barWidth, edgecolor='white', label='A4')
ax_00.bar(r6, A5, color=green_2, width=barWidth, edgecolor='white', label='A5')
ax_00.bar(r7, A6, color=green_3, width=barWidth, edgecolor='white', label='A6')
ax_00.set_xticks(np.arange(7) + barWidth*3.5)
#ax_00.set_xticklabels(('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'))
ax_00.set_ylabel('Sum BIC')
rc('text', usetex=True)
ax_00.text(-0.58, 4, r'$\times 10^4$', size=8, rotation=0, color='k', ha="center", va="center", fontname='serif')
rc('text', usetex=False)
ax_00.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax_00.set_xlabel('')

# Plot exceedance probabilites
ax_01 = plt.Subplot(f, gs01[2, 0])
f.add_subplot(ax_01)
ax_01.bar(r1, expr_A0, color='k', width=barWidth, edgecolor='white', label='A0')
ax_01.bar(r2, expr_A1, color=blue_1, width=barWidth, edgecolor='white', label='A1')
ax_01.bar(r3, expr_A2, color=blue_2, width=barWidth, edgecolor='white', label='A2')
ax_01.bar(r4, expr_A3, color=blue_3, width=barWidth, edgecolor='white', label='A3')
ax_01.bar(r5, expr_A4, color=green_1, width=barWidth, edgecolor='white', label='A4')
ax_01.bar(r6, expr_A5, color=green_2, width=barWidth, edgecolor='white', label='A5')
ax_01.bar(r7, expr_A6, color=green_3, width=barWidth, edgecolor='white', label='A6')
ax_01.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=False, ncol=7, framealpha=0.8)
ax_01.set_xticks(np.arange(7) + barWidth * 3.5)
ax_01.set_xticklabels(('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'))
ax_01.set_ylabel('pEP')

# --------------------------
# 4. Plot parameter recovery
# --------------------------

# Sigma parameter
# ---------------

gs04 = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs0[5:10, 0], hspace=0.1, wspace=1)
ax_02 = plt.Subplot(f, gs04[0, 0:2])
f.add_subplot(ax_02)
ax_02.set_xlim([-0.5, 2.5])
param_recov_sigma['x_pos'] = 1
ax_02 = sns.boxplot(x="x_pos", hue="which_param_sigma", y="sigma_bias", data=param_recov_sigma, notch=False,
                    showfliers=False, linewidth=0.8, width=0.2, boxprops=dict(alpha=1), ax=ax_02,
                    palette=sns.dark_palette("purple"))
ax_02.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=2)
ax_02.set_title('Sigma parameter')
ax_02.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax_02.set_ylabel('Estimation bias')
ax_02.set_xlabel('All agents')

# Lambda parameter
# ----------------

param_recov_3['x_pos'] = 0.5
param_recov_6['x_pos'] = 1.5
ax_11 = plt.Subplot(f, gs04[0, 2:4])
f.add_subplot(ax_11)
ax_11.set_xlim(-0.5, 2.5)
vertical_stack = pd.concat([param_recov_3, param_recov_6], axis=0, sort=False)
sns.boxplot(x="x_pos", hue="which_param_lambda", y="lambda_bias", data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_11,  palette=sns.dark_palette("purple"))
ax_11.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=2)
ax_11.set_title('Mixture parameter')
ax_11.set_xticklabels(('A3', 'A6'))
ax_11.set_xlabel('')
ax_11.set_ylabel('')


# Learning rate parameter
# -----------------------

ax_12 = plt.Subplot(f, gs04[0, 4:6])
f.add_subplot(ax_12)
param_recov_4['x_pos'] = 0
param_recov_5['x_pos'] = 1
param_recov_6['x_pos'] = 2
vertical_stack = pd.concat([param_recov_4, param_recov_5, param_recov_6], axis=0, sort=False)
sns.boxplot(x="x_pos", hue="which_param_alpha", y="alpha_bias", data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.6, boxprops=dict(alpha=1), ax=ax_12,  palette=sns.dark_palette("purple"))
ax_12.set_title('Learning rate parameter')
ax_12.set_xticklabels(('A4', 'A5', 'A6'))
ax_12.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=2)
ax_12.set_xlim([-0.5, 2.5])
ax_12.set_xlabel('')
ax_12.set_ylabel('')

# Choice noise parameter
# ----------------------

ax_10 = plt.Subplot(f, gs04[2, :])
f.add_subplot(ax_10)
param_recov_1['x_pos'] = 1
param_recov_2['x_pos'] = 2
param_recov_3['x_pos'] = 3
param_recov_4['x_pos'] = 4
param_recov_5['x_pos'] = 5
param_recov_6['x_pos'] = 6
vertical_stack = pd.concat([param_recov_1, param_recov_2, param_recov_3, param_recov_4, param_recov_5,
                            param_recov_6], axis=0, sort=False)
sns.boxplot(x="x_pos", hue="which_param_beta", y="beta_bias", data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_10, palette=sns.dark_palette("purple"))
ax_10.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
             fancybox=True, shadow=False, ncol=5)
ax_10.set_xticklabels(('A1', 'A2', 'A3', 'A4', 'A5', 'A6'))
ax_10.set_title('Choice noise parameter')
ax_10.set_ylabel('Estimation bias')
ax_10.set_xlabel('')

# Despine axes
sns.despine()

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', '', 'b', 'c', 'd', 'e', 'f', '']

# Add labels
label_subplots(f, texts)

# Save plot
savename = 'gb_figures/gb_figure_3.pdf'
plt.savefig(savename, dpi=400)

# Show figure
plt.show()
