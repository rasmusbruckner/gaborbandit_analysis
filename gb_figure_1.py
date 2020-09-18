""" Figure 1

    1. Load data and compute performance
    2. Run statistical tests
    3. Prepare figure
    4. Plot task trial schematic
    5. Plot task structure schematic
    6. Plot block example
    7. Plot performance
    8. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from latex_plt import latex_plt
from gb_plot_utils import cm2inch, plot_image, plot_arrow, label_subplots, plot_centered_text

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

# Economic decision making performance across trials
exp2_trial_part = exp2_data.groupby(['trial', 'participant'])['decision2.corr'].mean()
exp2_trial_mean = exp2_trial_part.mean(level=0)
exp2_trial_sd = exp2_trial_part.std(level=0)
exp2_trial_sem = exp2_trial_sd/np.sqrt(n_subj)

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

# Economic decision making performance across trials
exp3_trial_part = exp3_data.groupby(['trial', 'participant'])['decision2.corr'].mean()
exp3_trial_mean = exp3_trial_part.mean(level=0)
exp3_trial_sd = exp3_trial_part.std(level=0)
exp3_trial_sem = exp3_trial_sd/np.sqrt(n_subj)

# Save trial-wise performance of Exp 3 for Figure 4 (model-based results)
df_econ_trial = pd.DataFrame()
df_econ_trial['trial_mean'] = exp3_trial_mean
df_econ_trial['trial_sem'] = exp3_trial_sem
df_econ_trial['trial_sd'] = exp3_trial_sd
f = open('gb_data/trial_wise_part.pkl', 'wb')
pickle.dump(df_econ_trial, f)
f.close()

# Put variables in list for following bar plots
perc_mean_fig1 = [exp3_perc_mean, exp2_perc_mean]
perc_sem_fig1 = [exp3_perc_sem, exp2_perc_sem]
exp3_means = [exp3_econ_mean, exp3_perc_mean]
exp3_sd = [exp3_econ_sd, exp3_perc_sd]
exp3_sem = [exp3_econ_sem, exp3_perc_sem]

# ------------------------
# 2. Run statistical tests
# ------------------------

# 1. t-tests for perceptual decision making
# -----------------------------------------
t, p = stats.ttest_rel(exp2_perc_part, exp3_perc_part)
print('Perceptual decision making:\nt=%s, p=%s ' % (t, p))

# 2. t-tests for economic decision making
# -----------------------------------------
t, p = stats.ttest_rel(exp2_econ_part, exp3_econ_part)
print('Economic decision making:\nt=%s, p=%s ' % (t, p))

# -----------------
# 3. Prepare figure
# -----------------

# Figure properties
fig_ratio = 0.65
fig_witdh = 15
linewidth = 1
markersize = 2
fontsize = 6

# Create figure
f = plt.figure(figsize=cm2inch(fig_witdh, fig_ratio * fig_witdh))

# Create plot grid
gs0 = gridspec.GridSpec(18, 16, left=0.075, right=0.99, top=0.95, bottom=0.08, hspace=100, wspace=10)

# ----------------------------
# 4. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis
gs00 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs0[0:7, 0:8], wspace=0)
ax_0 = plt.Subplot(f, gs00[:])
f.add_subplot(ax_0)

# Picture paths
path = ['gb_figures/patches.png', 'gb_figures/fix_cross.png', 'gb_figures/fractals.png',
        'gb_figures/fix_cross.png', 'gb_figures/reward.png']

# Figure text
text = [r'$0.9-1.1 \ s.$', r'$1 \ s.$', r'$0.9-1.1 \ s.$', r'$1 \ s.$', r'$0.9-1.1 \ s.$']

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 0.2
image_y = 0.8

# Initialize text coordinates
text_y_dist = 0.075
text_pos = 'left_below'

# Cycle over images
for i in range(0, 5):

    # Plot images and text
    plot_image(f, path[i], cell_x0, cell_x1, image_y, ax_0, text_y_dist, text[i], text_pos, fontsize)

    # Update coordinates
    cell_x0 += 0.2
    cell_x1 += 0.2
    image_y += -0.1

# Delete unnecessary axes
ax_0.axis('off')

# ---------------------------------
# 5. Plot task structure schematic
# ---------------------------------

# Create subplot grid and axis
gs01 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs0[0:7, 9:16], wspace=0)
ax_1 = plt.Subplot(f, gs01[:])
f.add_subplot(ax_1)

# Define schematic coordinates
left_bound = 0.25
right_bound = 1
cell_width = right_bound - left_bound
center = left_bound + (cell_width / 2.0)
left_center = left_bound + (cell_width / 4.0)
right_center = right_bound - (cell_width / 4.0)

# Create schematic
# ----------------

# Left bound and states
ax_1.axvline(ymin=0, ymax=1, x=0.17, color='k', linewidth=0.5)
plot_centered_text(f, ax_1, left_bound, 0.98, center, 0.98, 'State 0', fontsize)
plot_centered_text(f, ax_1, center, 0.98, right_bound, 0.98, 'State 1', fontsize)

# Stage 1
cell_y0 = 0.6
cell_y1 = 0.9
ax_1, word_length, word_height, bbox = plot_centered_text(f, ax_1, -0.02, cell_y0, left_bound, cell_y1,
                                                          'Stage 1:\nPerceptual\ndecision', fontsize, c_type="y")
text_height = bbox.y1 - bbox.y0
plot_centered_text(f, ax_1, left_bound, cell_y0, center, cell_y1, 'Left stronger', fontsize)
plot_centered_text(f, ax_1, center, cell_y0, right_bound, cell_y1, 'Right stronger', fontsize)

# Stage 2 and 3
center_bias = 0.035  # shifts images toward center
text_post = 'centered_below'  # put text below image
text_y_dist = 0.35  # distance to lower bound of image
cell_y0 = 0.3  # lower position of image area
cell_y1 = 0.6  # upper position of image area
_, _, _, bbox = plot_centered_text(f, ax_1, -0.02, cell_y0, left_bound, cell_y1, 'Stage 2:\nEconomic\ndecision',
                                   fontsize, c_type="y")
plot_centered_text(f, ax_1, -0.02, 0.0, left_bound, cell_y0, 'Stage 3:\nReward', fontsize, c_type="y")
plot_centered_text(f, ax_1, 0.2, 0.12, left_bound, cell_y0, r'$\mu$', fontsize, c_type="y")
plot_centered_text(f, ax_1, 0.264, 0.14, left_bound, cell_y0, '=0.8:', fontsize, c_type="y")
plot_centered_text(f, ax_1, 0.2, -0.19, left_bound, cell_y0, r'$\mu$', fontsize, c_type="y")  #-0.175
plot_centered_text(f, ax_1, 0.264, -0.17, left_bound, cell_y0, '=0.2:', fontsize, c_type="y")

ax_1, bbox_image = plot_image(f, 'gb_figures/red_fractal.png', left_bound+center_bias, left_center+center_bias,
                              cell_y0, ax_1, text_y_dist, '80%\n\n20%', text_pos, fontsize, cell_y1=cell_y1)
plot_image(f, 'gb_figures/blue_fractal.png', left_center-center_bias, center-center_bias,
           cell_y0, ax_1, text_y_dist, '20%\n\n80%', text_pos, fontsize, cell_y1=cell_y1)
plot_image(f, 'gb_figures/red_fractal.png', center+center_bias, right_center+center_bias,
           cell_y0, ax_1, text_y_dist, '20%\n\n80%', text_pos, fontsize, cell_y1=cell_y1)
plot_image(f, 'gb_figures/blue_fractal.png', right_bound-center_bias, right_center-center_bias,
           cell_y0, ax_1, text_y_dist, '80%\n\n20%', text_pos, fontsize, cell_y1=cell_y1)

# Arrows
image_height = bbox_image.y1 - bbox_image.y0
image_center = bbox_image.y0 + image_height / 2.0
plot_arrow(ax_1, center, 0.95, center, bbox_image.y1)
plot_arrow(ax_1, center, bbox_image.y0, center, 0.2)
plot_arrow(ax_1, 0.12, image_center, 0.3, image_center)
ax_1.axis('off')

# ---------------------
# 6. Plot block example
# ---------------------

# Create subplot grid and axis
gs02 = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs0[7:18, 0:5], hspace=100, wspace=10)
ax_2 = plt.Subplot(f, gs02[0:2, 0])
f.add_subplot(ax_2)
ax_3 = plt.Subplot(f, gs02[2:4, 0])
f.add_subplot(ax_3)
ax_4 = plt.Subplot(f, gs02[4:6, 0])
f.add_subplot(ax_4)
ax_5 = plt.Subplot(f, gs02[6:8, 0])
f.add_subplot(ax_5)
ax_6 = plt.Subplot(f, gs02[8:10, 0])
f.add_subplot(ax_6)

# Use data of first participant
plot_data = exp3_data[exp3_data['id'] == 1].copy()

# Reset index
plot_data.loc[:, 'index'] = np.linspace(0, len(plot_data) - 1, len(plot_data))
plot_data = plot_data.set_index('index')
plot_data['a_t'] = plot_data['a_t']-1

# Determine number of plotted trials
n_t = 24

# Create x-axis
x = np.linspace(0, n_t, n_t+1)

# Adjust parameter for position of plot titles
plt.rcParams['axes.titlepad'] = 2

# State
ax_2.plot(x, plot_data['s_t'][:n_t], 'o', color='k', markersize=markersize)
ax_2.set_ylim([-0.2, 1.2])
ax_2.set_title('State')
ax_2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Presented contrast difference
ax_3.bar(x, plot_data['u_t'][:n_t], color='k', width=0.2)
ax_3.set_title('Contrast difference')
ax_3.set_ylim([-0.1, 0.1])
ax_3.tick_params(labelsize=fontsize)
ax_3.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax_3.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Perceptual decision
l2, = ax_4.plot(x, plot_data['d_t'][:n_t], 'o', color='k', markersize=markersize)
ax_4.set_ylim([-0.2, 1.2])
ax_4.set_title('Perceptual choice')
ax_4.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Economic decision
l3, = ax_5.plot(x[plot_data['a_t'][:n_t] == 0][:n_t], plot_data['a_t'][plot_data['a_t'] == 0][:n_t], 'o', color='k',
                markersize=markersize)
ax_5.plot(x[plot_data['a_t'][:n_t] == 1][:n_t], plot_data['a_t'][plot_data['a_t'] == 1][:n_t], 'o', color='k',
          markersize=markersize)
ax_5.set_ylim([-0.2, 1.2])
ax_5.set_title('Economic choice')
ax_5.set_ylim([-0.2, 1.2])
ax_5.tick_params(labelsize=fontsize)
ax_5.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Reward
l1, = ax_6.plot(x, plot_data['r_t'][:n_t], 'ko', markersize=markersize)
ax_6.set_ylim([-0.2, 1.2])
ax_6.tick_params(labelsize=fontsize)
ax_6.set_xlabel(r'Trial')
ax_6.set_title('Reward')

# -------------------
# 7. Plot performance
# -------------------

# Barplot of perceptual and economic choice performance
# -----------------------------------------------------

# Create subplot grid and axis
gs03 = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs0[7:18, 6:9], hspace=0, wspace=0)
ax_8 = plt.Subplot(f, gs03[0:4, 0])
f.add_subplot(ax_8)
ax_9 = plt.Subplot(f, gs03[6:10, 0])
f.add_subplot(ax_9)

# Perceptual choice
barlist = ax_8.bar([1, 2], perc_mean_fig1, yerr=perc_sem_fig1, alpha=1, edgecolor='k',
                   error_kw=dict(ecolor='k', linewidth=linewidth, capsize=5, capthick=1))
barlist[0].set_facecolor('#eeeeee')
barlist[1].set_facecolor('#eeeeee')

ax_8.set_ylabel('Perceptual-choice\nperformance')
ax_8.set_xticks([1, 2])
ax_8.set_xticklabels(['GB', 'Cont'])
ax_8.set_ylim([0.5, 1])
ax_8.set_xlabel(r'Condition')

# Economic choice
barlist = ax_9.bar([1, 2], exp3_means, yerr=exp3_sem, alpha=1, edgecolor='k',
                   error_kw=dict(ecolor='k', linewidth=linewidth, capsize=5, capthick=1))
barlist[0].set_facecolor('#eeeeee')
barlist[1].set_facecolor('#eeeeee')
ax_9.set_ylim([0.5, 0.9])
ax_9.set_xticks([1, 2])
ax_9.set_xticklabels(['GB', 'Cont'])
ax_9.set_ylabel('Economic-choice\nperformance')
ax_9.set_xlabel(r'Condition')

# Psychometric function
# ---------------------

# Create subplot grid and axis
gs04 = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs0[7:18, 10:16], hspace=0, wspace=0)
ax_10 = plt.Subplot(f, gs04[0:4, 0])
f.add_subplot(ax_10)
ax_7 = plt.Subplot(f, gs04[6:10, 0])
f.add_subplot(ax_7)

# Determine bins
bins = np.linspace(-0.08, 0.08, 18)

# Group contrasts into bins
cut = pd.cut(exp3_data['u_t'], bins)

# Group perceptual decisions by bin and id
d_t_binned = exp3_data['d_t'].groupby([cut, exp3_data['id']]).mean()

# Compute mean over participants and as a function of bin
d_t_binned_mean = d_t_binned.mean(level=0)

# Compute standard deviation over participants and as a function of bin
d_t_binned_sd = d_t_binned.std(level=0)

# Compute standard error of the mean as a function of bin
d_t_binned_sem = d_t_binned_sd/np.sqrt(n_subj)

# Plot psychometric function
x = np.linspace(-0.08, 0.08, 17)  # x-axis
ax_10.plot(x, d_t_binned_mean, color='k', linestyle='-', linewidth=linewidth)
ax_10.fill_between(x, d_t_binned_mean-d_t_binned_sem, d_t_binned_mean+d_t_binned_sem,
                   edgecolor='k', facecolor='k', alpha=0.2, linewidth=linewidth)
ax_10.set_ylabel('Frequency\nchoice right')
ax_10.set_xlabel('Contrast difference')
ax_10.set_ylim([0, 1])
plot_centered_text(f, ax_10, -0.08, -0.65, left_bound, cell_y0, 'Left', fontsize, c_type="y")
plot_centered_text(f, ax_10, 0.075, -0.65, left_bound, cell_y0, 'Right', fontsize, c_type="y")

# Economic decision making performance across trials
# --------------------------------------------------

# Create x-axis
x = np.linspace(1, 25, 25)

# Experiment 2
ax_7.plot(x, exp2_trial_mean, color='k', linestyle='--', linewidth=linewidth)
ax_7.fill_between(x, exp2_trial_mean-exp2_trial_sem, exp2_trial_mean+exp2_trial_sem,
                  edgecolor='k', facecolor='k', alpha=0.2, linewidth=linewidth)

# Experiment 3
ax_7.plot(x, exp3_trial_mean, color='k', linestyle='-', linewidth=linewidth)
ax_7.fill_between(x, exp3_trial_mean-exp3_trial_sem, exp3_trial_mean+exp3_trial_sem,
                  edgecolor='k', facecolor='k', alpha=0.2, linewidth=linewidth)

# Adjust plot properties
ax_7.set_xlabel(r'Trial')
ax_7.set_ylim([0.4, 1])
ax_7.legend(["Cont", "Exp"], loc=(0.7, 0.1))
ax_7.set_ylabel('Economic-choice\nperformance')

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b', 'c', '', '', '', '', 'd', 'f', 'e', 'g']

# Add labels
label_subplots(f, texts)

# Save plot
savename = 'gb_figures/gb_figure_1.pdf'
plt.savefig(savename, dpi=400)

plt.show()
