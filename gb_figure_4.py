""" Figure 4

    1. Load data prepare for plotting
    2. Prepare figure
    3. Plot posterior predictions
    4. Plot BIC based model comparison
    5. Plot exceedance-probability based model comparison
    6. Plot lambda parameter of each participant
    7. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import csv
from brokenaxes import brokenaxes
import os
from gb_plot_utils import cm2inch, label_subplots
import matplotlib.gridspec as gridspec
from latex_plt import latex_plt
from matplotlib import rc
# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# todo: cleaning:
# gb_postpred (done, außer 10 evals für prior noch klären)
# run_postpred soll das noch in ein object für simulationen mit task-agent-int etc.?
# todo: exceedance probs mit aktuellem file berechnen
# unused data entfernen

# ---------------------------------
# 1. Load data prepare for plotting
# ---------------------------------

# Parameter estimates
# These data were generated using the script gb_estimation.py
# part_params_bids = pd.read_pickle('gb_data/modelbased_bidstest.pkl')
# part_params_aufr = pd.read_pickle('gb_data/modelbased_aufraumtest_2.pkl')
part_params = pd.read_pickle('gb_data/modelbased_aufraumtest_2.pkl')
# bic_A6_aufr = part_params_aufr['A6_a_BIC']
# bic_A3_aufr = part_params_aufr['A3_a_BIC']
# bic_diff_aufr = bic_A6_aufr - bic_A3_aufr
# bic_diff_aufr = bic_diff_aufr.sort_values()
# bic_A6_bids = part_params_bids['A6_a_BIC']
# bic_A3_bids = part_params_bids['A3_a_BIC']
# bic_diff_bids = bic_A6_bids - bic_A3_bids
# bic_diff_bids = bic_diff_bids.sort_values()

# Extract BIC and compute BIC difference for the two best fitting models
bic_A6 = part_params['A6_a_BIC']
bic_A3 = part_params['A3_a_BIC']
bic_diff = bic_A6 - bic_A3

# Get all IDs
all_id = list(set(part_params['id']))

# Number of participants and blocks
N = len(all_id)

# Get BIC for each agent model
BIC_A0 = sum(part_params['A0_d_BIC']) + sum(part_params['A0_a_BIC'])
BIC_A1 = sum(part_params['A1_d_BIC']) + sum(part_params['A1_a_BIC'])
BIC_A2 = sum(part_params['A2_d_BIC']) + sum(part_params['A2_a_BIC'])
BIC_A3 = sum(part_params['A3_d_BIC']) + sum(part_params['A3_a_BIC'])
BIC_A4 = sum(part_params['A4_d_BIC']) + sum(part_params['A4_a_BIC'])
BIC_A5 = sum(part_params['A5_d_BIC']) + sum(part_params['A5_a_BIC'])
BIC_A6 = sum(part_params['A6_d_BIC']) + sum(part_params['A6_a_BIC'])
div = 10000  # set divisor to improve visibility in plot
BIC_A0 = BIC_A0 / div
BIC_A1 = BIC_A1 / div
BIC_A2 = BIC_A2 / div
BIC_A3 = BIC_A3 / div
BIC_A4 = BIC_A4 / div
BIC_A5 = BIC_A5 / div
BIC_A6 = BIC_A6 / div

# Participant exceedance probabilities
# ------------------------------------
# These data were generated in Matlab using SPM12
expr_part = list()
# with open("gb_data/exceedance_probs_part.csv") as csvfile:
with open("gb_data/exceedance_probs_part_april.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        # print(row)
        expr_part.append(row)
expr_part = [float(i) for i in expr_part[0]]

# Trial-wise average economic choice performance
# These data were generated using gb_figure_1.py
trial_wise_part = pd.read_pickle('gb_data/trial_wise_part.pkl')

# Model performance
# f = open('gb_data/postpred.pkl', 'rb')
# f = open('gb_data/postpred_april_beta.pkl', 'rb')
# These data were generated using gb_postpred.py
f = open('gb_data/postpred_final.pkl', 'rb')
post_pred = pickle.load(f)
f.close()

# Simulated performance
mean_corr_A0 = post_pred[0, :]
mean_corr_A1 = post_pred[1, :]
mean_corr_A2 = post_pred[2, :]
mean_corr_A3 = post_pred[3, :]
mean_corr_A4 = post_pred[4, :]
mean_corr_A5 = post_pred[5, :]
mean_corr_A6 = post_pred[6, :]

# -----------------
# 2. Prepare figure
# -----------------

# Figure properties
low_alpha = 0.3
medium_alpha = 0.6
high_alpha = 1
blue_1 = '#46b3e6'
blue_2 = '#4d80e4'
blue_3 = '#2e279d'
green_1 = '#94ed88'
green_2 = '#52d681'
green_3 = '#00ad7c'
fig_ratio = 0.65
fig_witdh = 15
fig_heigth = 10

# Create figure
f = plt.figure(figsize=cm2inch(fig_witdh, fig_heigth))

# Create plot grid
gs = gridspec.GridSpec(nrows=4, ncols=6, left=0.1, right=0.99, wspace=1.5, top=0.95, bottom=0.1, hspace=0.5)

# -----------------------------
# 3. Plot posterior predictions
# -----------------------------

# Prepare performance plot
gs01 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0:2, 0:6], wspace=4)
ax_2 = plt.Subplot(f, gs01[0, 2:6])
f.add_subplot(ax_2)
ax_2 = plt.gca()
ax_2.set_xlabel('Trial')
ax_2.set_ylabel('Performance')
ax_2.set_yticks(np.linspace(0.5, 0.9, 5))
ax_2.set_xlabel('Trial')
ax_2.set_ylabel('Economic choice performance')

# Plot posterior predictions
x = np.linspace(1, 25, 25)
a1 = ax_2.plot(x, np.mean(mean_corr_A1, 0), color=blue_1, linestyle='-', linewidth=2)
ax_2.plot(x, np.mean(mean_corr_A0, 0), color='black', linestyle='--', linewidth=2)
a2 = ax_2.plot(x, np.mean(mean_corr_A2, 0), color=blue_2, linestyle='-', linewidth=2)
ax_2.plot(x, trial_wise_part['trial_mean'], color='black', zorder=1, linewidth=3)
ax_2.fill_between(x, trial_wise_part['trial_mean']-trial_wise_part['trial_sem'],
                  trial_wise_part['trial_mean']+trial_wise_part['trial_sem'], facecolor='gray', alpha=low_alpha)
a3 = ax_2.plot(x, np.mean(mean_corr_A3, 0), color=blue_3, linestyle='-', linewidth=2)
ax_2.plot(x, np.mean(mean_corr_A4, 0), color=green_1, linestyle='-', linewidth=2)
ax_2.plot(x, np.mean(mean_corr_A5, 0), color=green_2, linestyle='-', linewidth=2)
ax_2.plot(x, np.mean(mean_corr_A6, 0), color=green_3, linestyle='-', linewidth=2)
part = Line2D([0], [0], color='black', lw=2)
A0 = Line2D([0], [0], color='black', lw=2, linestyle='--')
A1 = Line2D([0], [0], color=blue_1, lw=2)
A2 = Line2D([0], [0], color=blue_2, lw=2)
A3 = Line2D([0], [0], color=blue_3, lw=2)
A4 = Line2D([0], [0], color=green_1, lw=2)
A5 = Line2D([0], [0], color=green_2, lw=2)
A6 = Line2D([0], [0], color=green_3, lw=2)
ax_2.legend([part, A0, A1, A2, A3, A4, A5, A6],
            ['Behavioral Data', 'A0 | Random Choice', 'A1 | Normative belief-state',
             'A2 | Categorical belief-state', 'A3 | A1/A2 mixture model',
             'A4 | Belief-state Q-learning', 'A5 | Categorical Q-learning',
             'A6 | A4/A5 mixture model'], handler_map={tuple: HandlerTuple(ndivide=None)},
            bbox_to_anchor=[-0.2, 1], loc=0, borderaxespad=0, fontsize=7)

# ----------------------------------
# 4. Plot BIC based model comparison
# ----------------------------------

gs02 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[2:4, 0:6], wspace=2, hspace=0.5)
ax_0 = brokenaxes(ylims=((-2.2, -2.11), (-1.225, -1.05)), hspace=.15, d=0.01, subplot_spec=gs02[0, 0:2])

# Plot cumulated BIC's
ax_0.bar(0, BIC_A0, color='k', alpha=1, edgecolor='k')
ax_0.bar(1, BIC_A1, color=blue_1)
ax_0.bar(2, BIC_A2, color=blue_2)
ax_0.bar(3, BIC_A3, color=blue_3)
ax_0.bar(4, BIC_A4, color=green_1)
ax_0.bar(5, BIC_A5, color=green_2)
ax_0.bar(6, BIC_A6, color=green_3)
ax_0.set_ylabel('Sum BIC')
ax_0.set_xticks([0, 1, 2, 3, 4, 5, 6])
rc('text', usetex=True)
f.text(0.08, 0.51, r'$\times 10^4$', size=8, rotation=0, color='k', ha="center", va="center")
rc('text', usetex=False)
ax_0.set_xticklabels([''])

# Plot BIC difference between A3 and A6
ax_7 = plt.Subplot(f, gs02[0, 2:6])
f.add_subplot(ax_7)

# Initialize counter for participants in which A3 is better
counter = 0

# Cycle over participants
for i in range(0, N):

    if bic_diff[i] > 0:
        color = green_3
    else:
        color = blue_3
        counter += 1

    ax_7.bar([i], bic_diff[i], facecolor=color)

print('A3 better in %i out of %i' %(counter, N))
ax_7.set_ylabel('BIC A6-A3')
ax_7.set_xticklabels([''])

# -----------------------------------------------------
# 5. Plot exceedance-probability based model comparison
# -----------------------------------------------------

ax_1 = plt.Subplot(f, gs02[1, 0:2])
f.add_subplot(ax_1)
ax_1.bar(0, expr_part[0], color='k', alpha=high_alpha)
ax_1.bar(1, expr_part[1], color=blue_1)
ax_1.bar(2, expr_part[2], color=blue_2)
ax_1.bar(3, expr_part[3], color=blue_3)
ax_1.bar(4, expr_part[4], color=green_1)
ax_1.bar(5, expr_part[5], color=green_2)
ax_1.bar(6, expr_part[6], color=green_3)
ax_1.set_ylim(0, 1)
ax_1.set_ylabel('pEP')
ax_1.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax_1.set_xlabel('Agent')

# --------------------------------------------
# 6. Plot lambda parameter of each participant
# --------------------------------------------

ax_5 = plt.Subplot(f, gs02[1, 2:6])
f.add_subplot(ax_5)
lambda_mean = part_params['lambda_A3'].mean()
lambda_std = part_params['lambda_A3'].std()
lambda_sem = lambda_std/np.sqrt(N)
lambda_ordered = part_params['lambda_A3'].sort_values()
lambda_ordered = lambda_ordered.reset_index(drop=True)
ax_5.bar(np.arange(len(lambda_ordered)), lambda_ordered, facecolor=blue_3)
ax_5.set_xlabel('Participant')
ax_5.set_ylabel('Mixture\nparameter')
ax_5.axhline(part_params['lambda_A3'].mean(), color='k', lw=1.5, linestyle='--')
f.text(0.95, 0.27, 'Normative (A1)', size=5, rotation=0, color='k', ha="center", va="center")
f.text(0.49, 0.15, 'Categorical (A2)', size=5, rotation=0, color='k', ha="center", va="center")
print('Mean lambda: %f' %lambda_mean)


# ----------
# Pseudo r^2
# ----------

pseudo_r_sq_A3 = 1-(part_params['A3_llh']/part_params['A0_llh'])
pseudo_r_sq_A6 = 1-(part_params['A6_llh']/part_params['A0_llh'])
print('Pseudo r² A3: %f' % np.mean(pseudo_r_sq_A3))
print('Pseudo r² A6: %f' % np.mean(pseudo_r_sq_A6))
print('delta r²: %f' % (np.mean(pseudo_r_sq_A3)-np.mean(pseudo_r_sq_A6)))

# -------------------------------------------------------
# Average probability with which choices can be predicted
# --------------------------------------------------------

choice_pred_prob_A0 = np.mean(np.exp(-1*part_params['A0_llh']/300))
choice_pred_prob_A3 = np.mean(np.exp(-1*part_params['A3_llh']/300))
choice_pred_prob_A6 = np.mean(np.exp(-1*part_params['A6_llh']/300))
print('Average prediction accuracy A0: %f' %choice_pred_prob_A0)
print('Average prediction accuracy A3: %f' %choice_pred_prob_A3)
print('Average prediction accuracy A6: %f' %choice_pred_prob_A6)

# -------------------------------------
# 7. Add subplot labels and save figure
# -------------------------------------

# Despine and delete unnecessary axes of broken axes plot
sns.despine()
ax_0.axs[0].spines['bottom'].set_visible(False)
ax_0.big_ax.spines['left'].set_visible(False)

# Label letters
texts = [' a', ' ', ' ', 'b', 'c', 'd', 'e']

# Add labels
label_subplots(f, texts)

# Save figure
savename = 'gb_figures/gb_figure_4.pdf'
plt.savefig(savename, transparent=True)

plt.show()
