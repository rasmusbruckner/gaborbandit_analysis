""" SM Figure 1

    1. Load data
    2. Prepare figure
    3. Plot task statistics
    4. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
from gb_plot_utils import cm2inch, label_subplots
import seaborn as sns
from latex_plt import latex_plt
from pathlib import Path, PureWindowsPath
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# -----------------------------------------------------
# 1. Load data and compute experimental task statistics
# -----------------------------------------------------

# Data of third experiment
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Extract all blocks of our participants
all_blocks = list(set(exp3_data['block_file']))

# Adjust path to gb task
path = '/Users/rasmus/Dropbox/gabor_bandit/code/gaborbandit_task/'
os.chdir(path)

# Load data of block 0
filename = PureWindowsPath(all_blocks[0])
# ... and change to pure Mac path...
hh = Path(filename)
# ... and load data
blocks = pd.read_excel(hh)

# Initialize new data with block 0
new_data = blocks

# Cycle over blocks
for i in range(1, len(all_blocks)):

    # Repeat for all blocks and append to blocks data frame
    filename = PureWindowsPath(all_blocks[i])
    hh = Path(filename)
    new_data = pd.read_excel(hh)

    blocks = blocks.append(new_data, ignore_index=True)

# Compute reward frequency
contingency = np.sum(blocks['outcomes'] == 1) / len(blocks)

# Compute mean absolute contrast difference
abs_diff = np.mean(blocks['PU'])

# Compute frequency red fractal up
up = sum(blocks['redFractal'] == 'up') / len(blocks)

# Compute frequency target patch left
left = sum(blocks['targetPatch'] == 'left') / len(blocks)

# -----------------
# 2. Prepare figure
# -----------------

# Figure properties
fig_witdh = 15
fig_height = 7.5

# Create figure
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# Create plot grid
gs0 = gridspec.GridSpec(1, 1)

# Create subplot grid
gs00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0], wspace=0.75)

# Plot reward frequency
ax = plt.Subplot(f, gs00[0, 0])
f.add_subplot(ax)
ax.bar(0, contingency, color='k')
ax.set_ylabel('Reward frequency')
ax.tick_params(
    axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot mean absolute contrast difference
ax = plt.Subplot(f, gs00[0, 1])
f.add_subplot(ax)
ax.bar(0, abs_diff, color='k')
ax.set_ylabel('Mean absolute contrast difference')
ax.tick_params(
    axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot frequency red fractal up
ax = plt.Subplot(f, gs00[0, 2])
f.add_subplot(ax)
ax.bar(0, up, color='k')
ax.set_ylabel('Frequency high/low contrast patch left')
ax.tick_params(
    axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Plot frequency target patch left
ax = plt.Subplot(f, gs00[0, 3])
f.add_subplot(ax)
ax.set_ylabel('Frequency red fractal up')
ax.tick_params(
    axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.bar(0, left, color='k')

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b ', 'c ', 'd']

# Add labels
label_subplots(f, texts, x_offset=0.075)

# Delete unnecessary axes
sns.despine()

# Save plot
savename = '/Users/rasmus/Dropbox/gabor_bandit/code/gaborbandit_analysis/gb_figures/gb_sm_figure_1.pdf'
plt.savefig(savename, dpi=400)

# Show plot
plt.show()
