""" This script plots SM Figure 3

    1. Prepare figure
    2. Plot example of no perceptual uncertainty
    3. Plot example of high perceptual uncertainty
    4. Plot example of maximal perceptual uncertainty
    5. Add subplot labels and save figure

"""

import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
from truncate_colormap import truncate_colormap
from latex_plt import latex_plt
import matplotlib.pyplot as plt
from gb_plot_utils import label_subplots, plot_pmu
from GbAgentVars import AgentVars
from GbAgent import Agent
import os

# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# -----------------
# 1. Prepare figure
# -----------------

# Create figure with multiple subplots
f = plt.figure(figsize=(6.4, 2.4))
ax_0 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)

# Define colormap
cmap = truncate_colormap(plt.get_cmap('bone'), 0.1, 0.9)

# Create range of colors for range of contrast differences
cval_ind = range(10)
c_norm = colors.Normalize(vmin=0, vmax=cval_ind[-1])
scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)

# --------------------------------------------
# 2. Plot example of no perceptual uncertainty
# --------------------------------------------

# Create agent object
agent_vars = AgentVars()
agent_vars.agent = 1
agent = Agent(agent_vars)

# Example of \pi_0 = 1, \pi_1 = 0
agent, ax_0, v_a_t_hs, _, _, _ = plot_pmu(agent, 1, 0, cval_ind, scalar_map, ax_0)

# Adjust properties of the plots
ax_0.set_ylabel(r'$p_t(\mu)$')
ax_0.set_xlabel(r'$\mu$')
ax_0.set_title('No perceptual uncertainty')
ax_0.set_ylim([-0.1, 10.1])

# ----------------------------------------------
# 3. Plot example of high perceptual uncertainty
# ----------------------------------------------

# Create subplot grid
ax_1 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)

# Example of \pi_0 = 1, \pi_1 = 0
agent, ax_1, v_a_t_hs, _, _, _ = plot_pmu(agent, 0.6, 0.4, cval_ind, scalar_map, ax_1, plot_leg=False)

# Adjust properties of the plots
ax_1.set_ylabel(r'$p_t(\mu)$')
ax_1.set_xlabel(r'$\mu$')
ax_1.set_title('High perceptual uncertainty')
ax_1.set_ylim([-0.1, 10.1])

# -------------------------------------------------
# 4. Plot example of maximal perceptual uncertainty
# -------------------------------------------------

# Create subplot grid
ax_2 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)

# Example of \pi_0 = 1, \pi_1 = 0
agent, ax_2, v_a_t_hs, _, _, _ = plot_pmu(agent, 0.5, 0.5, cval_ind, scalar_map, ax_2, plot_leg=False)

# Adjust properties of the plots
ax_2.set_ylabel(r'$p_t(\mu)$')
ax_2.set_xlabel(r'$\mu$')
ax_2.set_title('Maximal perceptual uncertainty')
ax_2.set_ylim([-0.1, 10.1])

# Use figure space more efficiently
plt.tight_layout()
sns.despine()

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b', 'c']

# Add labels
label_subplots(f, texts)

savename = 'gb_figures/gb_sm_figure_3.pdf'
plt.savefig(savename, transparent=True)

# Show figure
plt.show()
