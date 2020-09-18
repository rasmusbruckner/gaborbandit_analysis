""" This script plots Figure 2

    1. Prepare plotting
    2. Plot task-agent-interaction schematic
    3. Plot belief state illustration
    4. Plot agent table
    5. Plot agent performance comparison
    6. Agent demonstration across task block
    7. Add subplot labels and save figure
"""

import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
import pickle
from truncate_colormap import truncate_colormap
from latex_plt import latex_plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from gb_plot_utils import plot_arrow, label_subplots, cm2inch, bs_plot, plot_rec, plot_image, plot_table,\
    plot_centered_text, center_x, get_text_coords, plot_agent_demo, plot_observations, plot_pmu
from GbAgentVars import AgentVars
from GbAgent import Agent
from gb_simulation import gb_simulation
import matplotlib.gridspec as gridspec
import os


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# -------------------
# 1. Prepare plotting
# -------------------

# Load model simulation data
# --------------------------

# Model performance
f = open('gb_data/predictions.pkl', 'rb')  # these simulation data were generated using the script gb_postpred.py
pred = pickle.load(f)
f.close()

# Simulations
mean_corr_A0 = pred[0, :]
mean_corr_A1 = pred[1, :]
mean_corr_A2 = pred[2, :]
mean_corr_A3 = pred[3, :]
mean_corr_A4 = pred[4, :]
mean_corr_A5 = pred[5, :]
mean_corr_A6 = pred[6, :]

# Plot properties
markersize = 2
fontsize = 6
fig_ratio = 1.4
fig_witdh = 15
fig_height = 17

# Define figure
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))
gs0 = gridspec.GridSpec(nrows=7, ncols=2, left=0.085, hspace=0.6, top=.95, right=0.99, wspace=0.3, bottom=0.05)

# ----------------------------------------
# 2. Plot task-agent-interaction schematic
# ----------------------------------------

gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0:2, 0])
ax_00 = plt.Subplot(f, gs01[0, 0])
f.add_subplot(ax_00)

# Agent box
# ---------

# Plot upper rectangle
cell_x0, width = 0, 1
cell_y0, height = 0.55, 0.45
ax_00 = plot_rec(ax_00, patches, cell_x0, width, cell_y0, height)

# Plot lower rectangle
cell_y0, height = 0, 0.45
ax_00 = plot_rec(ax_00, patches, cell_x0, width, cell_y0, height)

# Plot connecting arrows
x1, y1 = 0, 0.2
x2, y2 = 0, 0.8
ax_00 = plot_arrow(ax_00, x1, y1, x2, y2, shrink_a=0, shrink_b=0, connectionstyle="bar,fraction=0.1", arrow_style="<-")
x1 = 1
x2 = 1
ax_00 = plot_arrow(ax_00, x1, y1, x2, y2, shrink_a=0, shrink_b=0, connectionstyle="bar,fraction=-0.1", arrow_style="->")

# Add agent text
printed_word = "Agent"
cell_x0 = 0
cell_y0 = 0
word_length, _, _ = get_text_coords(f, ax_00, cell_x0, cell_y0, printed_word, fontsize)
x = center_x(0, 1, word_length, correct_for_length=True)
ax_00.text(x, 0.92, printed_word)

# Plot smaller left rectangle with text
cell_x0, width = 0.025, 0.3
cell_y0, height = .6, .2
ax_00 = plot_rec(ax_00, patches, cell_x0, width, cell_y0, height)
cell_x1 = cell_x0 + width
cell_y1 = cell_y0 + height
plot_centered_text(f, ax_00, cell_x0, cell_y0, cell_x1, cell_y1,
                   'Belief state,\nperceptual\ndecision', fontsize)
printed_word = "Perception"
word_length, _, _ = get_text_coords(f, ax_00, cell_x0, cell_y0, printed_word, fontsize)
x = center_x(cell_x0, width, word_length, correct_for_length=True)
ax_00.text(x, 0.82, printed_word)

# Plot smaller middle rectangle with text
cell_x0, width = 0.025*2 + 0.3,  0.3
ax_00 = plot_rec(ax_00, patches, cell_x0, width, cell_y0, height)
cell_x1 = cell_x0 + width
cell_y1 = cell_y0 + height
plot_centered_text(f, ax_00, cell_x0, cell_y0, cell_x1, cell_y1,
                   'Action-depen-\ndent expected\nvalue', fontsize)
printed_word = "Action"
word_length, _, _ = get_text_coords(f, ax_00, cell_x0, cell_y0, printed_word, fontsize)
x = center_x(cell_x0, width, word_length, correct_for_length=True)
ax_00.text(x, 0.82, printed_word)

# Plot smaller right rectangle with text
cell_x0, width = 0.025*3 + 0.3*2, 0.3
ax_00 = plot_rec(ax_00, patches, cell_x0, width, cell_y0, height)
cell_x1 = cell_x0 + width
cell_y1 = cell_y0 + height
plot_centered_text(f, ax_00, cell_x0, cell_y0, cell_x1, cell_y1,
                   'Reward-based\nlearning', fontsize)
printed_word = "Learning"
word_length, _, _ = get_text_coords(f, ax_00, cell_x0, cell_y0, printed_word, fontsize)
x = center_x(cell_x0, width, word_length, correct_for_length=True)
ax_00.text(x, 0.82, printed_word)

# Task box
# --------

# Add task text
printed_word = "Task"
word_length, _, _ = get_text_coords(f, ax_00, cell_x0, cell_y0, printed_word, fontsize)
x = center_x(0, 1, word_length, correct_for_length=True)
ax_00.text(x, 0.38, printed_word)

# Add task images and text
img_path = 'gb_figures/patches.png'
cell_x0 = 0.025
cell_x1 = 0.325
image_y = 0.15
text_y_dist = 0.025
text_pos = 'above'
text = 'Patches'
plot_image(f, img_path, cell_x0, cell_x1, image_y, ax_00, text_y_dist, text, text_pos, fontsize, zoom=0.16)
img_path = 'gb_figures/fractals.png'
cell_x0 = 0.35
cell_x1 = 0.65
text = 'Fractals'
plot_image(f, img_path, cell_x0, cell_x1, image_y, ax_00, text_y_dist, text, text_pos, fontsize, zoom=0.16)
img_path = 'gb_figures/reward.png'
cell_x0 = 0.675
cell_x1 = 0.975
text = 'Reward'
plot_image(f, img_path, cell_x0, cell_x1, image_y, ax_00, text_y_dist, text, text_pos, fontsize, zoom=0.16)

# Turn unnecessary axes off
ax_00.axis('off')

# ---------------------------------
# 3. Plot belief state illustration
# ---------------------------------

# Define subplots
# ---------------
gs00 = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs0[0:2, 1])
ax_01 = plt.Subplot(f, gs00[0, 0])
f.add_subplot(ax_01)

ax_02 = plt.Subplot(f, gs00[0, 1])
f.add_subplot(ax_02)

ax_03 = plt.Subplot(f, gs00[1:3, 0])
f.add_subplot(ax_03)

ax_04 = plt.Subplot(f, gs00[1:3, 1])
f.add_subplot(ax_04)

ax_05 = plt.Subplot(f, gs00[3:5, 0])
f.add_subplot(ax_05)

ax_06 = plt.Subplot(f, gs00[3:5, 1])
f.add_subplot(ax_06)

# Set perceptual sensitivity conditions
hs = 0.02
ms = 0.04
ls = 0.06

# Call AgentVars and Agent objects
agent_vars = AgentVars()
agent_vars.agent = 1
agent_vars.task_agent_analysis = False
agent = Agent(agent_vars)

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

# Create range of colors for range of contrast differences
cval_ind = range(len(d_t))

# Set resolution for distributions over observatiosn
x = np.linspace(-x_lim, x_lim, 1000)

# Initialize belief-state arrays
pi1_hs = np.zeros(24)  # state 1 low noise
pi1_ms = np.zeros(24)  # state 1 high noise

# Define colormap
cmap = truncate_colormap(plt.get_cmap('bone'), 0, 0.7)

# Create range of colors for range of contrast differences
cval_ind = range(len(d_t))
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
# Define colormap

# Plot contrast differences and observation probabilities
ax_01, ax_03 = plot_observations(ax_01, ax_03, scalar_map, d_t, cval_ind, hs, x, x_lim)

# Belief state plot
ax_05 = bs_plot(ax_05, agent, U, cval_ind, hs, scalar_map)
which_state = 1
ax_05 = bs_plot(ax_05, agent, U, cval_ind, hs, scalar_map, which_state)
ax_05.set_ylim(-0.1, 1.1)
ax_05.tick_params(labelsize=fontsize)
ax_05.set_xlim(-x_lim, x_lim)
ax_05.set_ylabel('Belief\nstate', fontsize=fontsize)
ax_05.set_xlabel(r'Contrast difference', fontsize=fontsize)

# Plot contrast differences and observation probabilities
ax_02, ax_04 = plot_observations(ax_02, ax_04, scalar_map, d_t, cval_ind, ms, x, x_lim, labels="reduced")

# Belief state plot
ax_06 = bs_plot(ax_06, agent, U, cval_ind, ms, scalar_map)
which_state = 1
ax_06 = bs_plot(ax_06, agent, U, cval_ind, ms, scalar_map, which_state)
ax_06.set_ylim(-0.1, 1.1)
ax_06.tick_params(labelsize=fontsize)
ax_06.set_xlim(-x_lim, x_lim)
ax_06.set_xlabel(r'Contrast difference', fontsize=fontsize)
ax_06.axes.get_yaxis().set_ticks([])

# --------------------
# 4. Plot agent table
# --------------------

# Define subplots
# ---------------
gs02 = gridspec.GridSpecFromSubplotSpec(10, 3, subplot_spec=gs0[2:4, 0:3], wspace=0.5)
ax_07 = plt.Subplot(f, gs02[0:10, 0:2])
f.add_subplot(ax_07)

# XX
row_lines, col_lines = plot_table(ax_07)
ax_07.set_ylim(0, 1)
ax_07.set_xlim(0, 1)

# Headers
col_headers = ['Agent', 'Perceptual choice', 'Economic choice EV', 'Learning']
row_headers = ['A6', 'A5',  'A4', 'A3', 'A2', 'A1', 'A0']

# Table content
data_names = ['High belief state', 'High belief state', 'High belief state', 'High belief state', 'High belief state', 'High belief state', 'Random',
              'A4/A5 mixture', 'Categorical', 'Belief-state weighted', 'A1/A2 mixture', 'Categorical', 'Belief-state weighted', 'Random',
              'A4/A5 Q-learning mixture', 'Categorical Q-learning', 'Belief-state Q-learning',
              'A1/A2 Bayesian mixture', 'Categorical Bayesian', 'Belief-state Bayesian', 'Random']

# Set table fontsize
fontsize = 5

# Initialize counter
counter = 0

# Cycle over columns and rows
for i in range(1, len(col_lines)-1):

    for j in range(0, len(row_lines)-2):

        # Plot text in center of cells
        plot_centered_text(f, ax_07, col_lines[i], row_lines[j], col_lines[i+1], row_lines[j+1], data_names[counter],
                           fontsize)

        # Update counter
        counter += 1

# Adjust font weight for headers
fontweight = 'bold'

for i in range(0, len(row_headers)):

    plot_centered_text(f, ax_07, col_lines[0], row_lines[i], col_lines[1], row_lines[i+1], row_headers[i], fontsize,
                       fontweight=fontweight)


for i in range(0, len(col_headers)):

    plot_centered_text(f, ax_07, col_lines[i], row_lines[7], col_lines[i+1], row_lines[8], col_headers[i], fontsize,
                       fontweight=fontweight)

# Turn axis off
ax_07.axis('off')

# ------------------------------------
# 5. Plot agent performance comparison
# ------------------------------------

# Define subplot
ax_08 = plt.Subplot(f, gs02[0:8, 2])
f.add_subplot(ax_08)

# Set plot parameters
low_alpha = 0.3
medium_alpha = 0.6
high_alpha = 1
blue_1 = '#46b3e6'
blue_2 = '#4d80e4'
blue_3 = '#2e279d'
green_1 = '#94ed88'
green_2 = '#52d681'
green_3 = '#00ad7c'

# Plot performance
ax_08.bar(0, np.mean(mean_corr_A0), color='k')
ax_08.bar(1, np.mean(mean_corr_A1), color=blue_1)
ax_08.bar(2, np.mean(mean_corr_A2), color=blue_2)
ax_08.bar(3, np.mean(mean_corr_A3), color=blue_3)
ax_08.bar(4, np.mean(mean_corr_A4), color=green_1)
ax_08.bar(5, np.mean(mean_corr_A5), color=green_2)
ax_08.bar(6, np.mean(mean_corr_A6), color=green_3)
ax_08.set_xticks([0, 1, 2, 3, 4, 5, 6])
plt.ylim(0.4, 0.8)
plt.ylabel('Economic-choice\nperformance')
plt.xlabel('Agent')

# ----------------------------------------
# 6. Agent demonstration across task block
# ----------------------------------------

# Define subplots
# ---------------
gs03 = gridspec.GridSpecFromSubplotSpec(10, 4, subplot_spec=gs0[4:7, 0:3], wspace=0.1, hspace=1)
ax_09 = plt.Subplot(f, gs03[0:2, 0])
f.add_subplot(ax_09)
ax_10 = plt.Subplot(f, gs03[2:4, 0])
f.add_subplot(ax_10)
ax_11 = plt.Subplot(f, gs03[4:6, 0])
f.add_subplot(ax_11)
ax_12 = plt.Subplot(f, gs03[6:8, 0])
f.add_subplot(ax_12)
ax_13 = plt.Subplot(f, gs03[8:10, 0])
f.add_subplot(ax_13)

ax_14 = plt.Subplot(f, gs03[0:2, 1])
f.add_subplot(ax_14)
ax_15 = plt.Subplot(f, gs03[2:4, 1])
f.add_subplot(ax_15)
ax_16 = plt.Subplot(f, gs03[4:6, 1])
f.add_subplot(ax_16)
ax_17 = plt.Subplot(f, gs03[6:8, 1])
f.add_subplot(ax_17)
ax_18 = plt.Subplot(f, gs03[8:10, 1])
f.add_subplot(ax_18)

ax_19 = plt.Subplot(f, gs03[0:2, 2])
f.add_subplot(ax_19)
ax_20 = plt.Subplot(f, gs03[2:4, 2])
f.add_subplot(ax_20)
ax_21 = plt.Subplot(f, gs03[4:6, 2])
f.add_subplot(ax_21)
ax_22 = plt.Subplot(f, gs03[6:8, 2])
f.add_subplot(ax_22)
ax_23 = plt.Subplot(f, gs03[8:10, 2])
f.add_subplot(ax_23)

ax_24 = plt.Subplot(f, gs03[0:2, 3])
f.add_subplot(ax_24)
ax_25 = plt.Subplot(f, gs03[2:4, 3])
f.add_subplot(ax_25)
ax_26 = plt.Subplot(f, gs03[4:6, 3])
f.add_subplot(ax_26)
ax_27 = plt.Subplot(f, gs03[6:8, 3])
f.add_subplot(ax_27)
ax_28 = plt.Subplot(f, gs03[8:10, 3])
f.add_subplot(ax_28)

# Set simulation parameters
T = 25
B = 10
sigma = 0.04
beta = 100

# Define x-axis
x = np.linspace(0, 24, 25)

# Simulate data with agent 1
np.random.seed(123)
agent = 1
df = gb_simulation(T, B, sigma, agent, beta)
header = "Belief-state Bayesian A1"
plot_agent_demo(ax_09, ax_10, ax_11, ax_12, ax_13, x, df, header, blue_1, markersize=markersize, fontsize=fontsize)

# Simulate data with agent 2
np.random.seed(123)
agent = 2
df = gb_simulation(T, B, sigma, agent, beta)
header = "Categorical Bayesian A2"
plot_agent_demo(ax_14, ax_15, ax_16, ax_17, ax_18, x, df, header, blue_2, markersize=markersize,
                fontsize=fontsize, labels='reduced', bs='cat')

# Simulate data with agent 4
np.random.seed(123)
agent = 4
df = gb_simulation(T, B, sigma, agent, beta)
header = "Belief-state Q-learning A4"
plot_agent_demo(ax_19, ax_20, ax_21, ax_22, ax_23, x, df, header, green_1, markersize=markersize,
                fontsize=fontsize, labels='reduced')

# Simulate data with agent 5
np.random.seed(123)
agent = 5
df = gb_simulation(T, B, sigma, agent, beta)
header = "Categorical Q-learning A5"
plot_agent_demo(ax_24, ax_25, ax_26, ax_27, ax_28, x, df, header, green_3, markersize=markersize,
                fontsize=fontsize, labels='reduced', bs='cat')

# Turn unnecessary axes off
sns.despine()

# -------------------------------------
# 7. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b', '', '', '', '', '', 'c', 'd', 'e', '']

# Add labels
label_subplots(f, texts)

# Save plot
savename = 'gb_figures/gb_figure_2.pdf'
plt.savefig(savename, dpi=400)

# Show figure
plt.show()
