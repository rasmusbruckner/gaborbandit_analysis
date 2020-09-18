""" Figure 1

    1. Load data and compute performance
    2. SM Figure 2: Single participant performances
    3. Save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from latex_plt import latex_plt
from gb_plot_utils import cm2inch, label_subplots

# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# ------------------------------------
# 1. Load data and compute performance
# ------------------------------------

# Load preprocessed data of all experiments
exp1_data = pd.read_pickle('gb_data/gb_exp1_data.pkl')
exp2_data = pd.read_pickle('gb_data/gb_exp2_data.pkl')
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Get all IDs
all_id = list(set(exp1_data['participant']))

# Number of participants and blocks
n_subj = len(all_id)
n_blocks2 = len(list(set(exp2_data['blockNumber'])))
n_blocks3 = len(list(set(exp3_data['blockNumber'])))

# Add trial number to data frames
exp2_data.loc[:, 'trial'] = np.tile(np.linspace(0, 24, 25), [len(all_id)*n_blocks2])
exp3_data.loc[:, 'trial'] = np.tile(np.linspace(0, 24, 25), [len(all_id)*n_blocks3])

# Experiment 2
# ------------
# Mean perceptual decision making performance
exp2_perc_part = exp2_data.groupby(['participant'])['decision1.corr'].mean()
exp2_perc_mean = np.mean(exp2_perc_part)
exp2_perc_sd = np.std(exp2_perc_part)
exp2_perc_sem = exp2_perc_sd/np.sqrt(n_subj)

# Mean economic decision making performance
exp2_econ_part = exp2_data.groupby(['participant'])['decision2.corr'].mean()
exp2_econ_mean = np.mean(exp2_econ_part)
exp2_econ_sd = np.std(exp2_econ_part)
exp2_econ_sem = exp2_econ_sd/np.sqrt(n_subj)

# Experiment 3
# ------------
# Mean perceptual decision making performance
exp3_perc_part = exp3_data.groupby(['participant'])['decision1.corr'].mean()
exp3_perc_mean = np.mean(exp3_perc_part)
exp3_perc_sd = np.std(exp3_perc_part)
exp3_perc_sem = exp3_perc_sd/np.sqrt(n_subj)

# Mean economic decision making performance
exp3_econ_part = exp3_data.groupby(['participant'])['decision2.corr'].mean()
exp3_econ_mean = np.mean(exp3_econ_part)
exp3_econ_sd = np.std(exp3_econ_part)
exp3_econ_sem = exp3_econ_sd/np.sqrt(n_subj)

# --------------
# 2. SM Figure 2
# --------------

# Mean perceptual decision making performance
exp1_perc_part = exp1_data.groupby(['participant'])['decision1.corr'].mean()
exp1_perc_mean = np.mean(exp1_perc_part)
exp1_perc_sd = np.std(exp1_perc_part)
exp1_perc_sem = exp1_perc_sd/np.sqrt(n_subj)

# Figure properties
fig_ratio = 0.65
fig_witdh = 15
linewidth = 1
markersize = 2
fontsize = 6

# Create figure
f = plt.figure(figsize=cm2inch(fig_witdh, fig_ratio * fig_witdh))

# Prepare figure
ax_1 = plt.subplot2grid((3, 2), (0, 0))
ax_2 = plt.subplot2grid((3, 2), (1, 0))
ax_3 = plt.subplot2grid((3, 2), (2, 0))
ax_4 = plt.subplot2grid((3, 2), (1, 1))
ax_5 = plt.subplot2grid((3, 2), (2, 1))

# Experiment 1
# ------------
# Mean perceptual decision making performance
ax_1.bar(np.arange(len(all_id)), exp3_perc_part, color='k', alpha=0.2)
ax_1.axhline(exp1_perc_mean, color='black', lw=0.5, linestyle='--')
ax_1.set_ylim([0.5, 1])
ax_1.set_title('Perceptual decision making')
ax_1.set_xlabel('Participant')
ax_1.set_ylabel('Perceptual-choice\nperformance')

# Experiment 2
# ------------
# Mean perceptual decision making performance
ax_2.bar(np.arange(len(all_id)), exp2_perc_part, color='k', alpha=0.2)
ax_2.axhline(exp2_perc_mean, color='black', lw=0.5, linestyle='--')
ax_2.set_ylim([0.5, 1])
ax_2.set_title('Control task')
ax_2.set_xlabel('Participant')
ax_2.set_ylabel('Perceptual-choice\nperformance')

# Mean economic decision making performance
ax_4.bar(np.arange(len(all_id)), exp2_econ_part, color='k', alpha=0.2)
ax_4.axhline(exp2_econ_mean, color='black', lw=0.5, linestyle='--')
ax_4.set_ylim([0.5, 1])
ax_4.set_title('Control task')
ax_4.set_ylabel('Economic-choice\nperformance')
ax_4.set_xlabel('Participant')

# Experiment 3
# ------------
# Mean perceptual decision making performance
ax_3.bar(np.arange(len(all_id)), exp3_perc_part, color='k', alpha=0.2)
ax_3.axhline(exp3_perc_mean, color='black', lw=0.5, linestyle='--')
ax_3.set_ylim([0.5, 1])
ax_3.set_title('Gabor-bandit task')
ax_3.set_xlabel('Participant')
ax_3.set_ylabel('Perceptual-choice\nperformance')

# Mean economic decision making performance
ax_5.bar(np.arange(len(all_id)), exp3_econ_part, color='k', alpha=0.2)
ax_5.axhline(exp3_econ_mean, color='black', lw=0.5, linestyle='--')
ax_5.set_ylim([0.5, 1])
ax_5.set_title('Gabor-bandit task')
ax_5.set_ylabel('Economic-choice\nperformance')
ax_5.set_xlabel('Participant')

# Adjust figure space
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.10, right=0.9, hspace=1, wspace=0.35)
sns.despine()

# --------------
# 3. Save figure
# --------------

# Label letters
texts = ['a', 'b', 'd', 'c', 'e']

# Add labels
label_subplots(f, texts, x_offset=0.06, y_offset=0.015)

savename = 'gb_figures/gb_sm_figure_2.pdf'
plt.savefig(savename)

plt.show()
