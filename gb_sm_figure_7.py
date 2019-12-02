# This script plots Figure SM 7

from scipy import stats
import matplotlib
import seaborn as sns
from latex_plt import latex_plt
import matplotlib.pyplot as plt
import numpy as np
from gb_plot_utils import label_subplots, cm2inch
import os
from matplotlib import rc
# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)
# Activate latex
rc('text', usetex=True)

# Figure properties
fig_width = 15
fig_height = 10

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create subplot axes
ax_0 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
ax_1 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
ax_2 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
ax_3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
ax_4 = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)
ax_5 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)

# Range of observations
U = np.linspace(-0.15, 0.15, 100)

# Example 1: Strong negative contrast difference
mu = -0.1
sigma = 0.02
p_o_giv_u = stats.norm.pdf(U, mu, sigma)
val = [stats.norm.cdf(0, mu, sigma), 1 - stats.norm.cdf(0, mu, sigma)]
ax_0.plot(U, p_o_giv_u, color='k')
ax_0.axvline(x=0, color='k', alpha = 0.5, linewidth = 1, linestyle = '--')
ax_0.set_xlabel(r'$c_t$')
ax_0.set_ylabel(r'$p^{\sigma^2}(o_t|c_t=-0.1)$')
ax_3.bar([0,1], val,color='k')
ax_3.set_ylim([0, 1])
ax_3.set_ylabel(r'$p(d_t|c_t=0.1)$')
ax_3.set_xlabel(r'$d_t$')
ax_3.set_xticks([0, 1])
ax_3.set_xticks([0, 1])

# Example 2: Weak negative contrast difference
mu = -0.01
sigma = 0.02
p_o_giv_u = stats.norm.pdf(U, mu, sigma)
val = [stats.norm.cdf(0, mu, sigma), 1 - stats.norm.cdf(0, mu, sigma)]#0.5
ax_1.plot(U, p_o_giv_u, color='k')
ax_1.axvline(x=0, color='k', alpha = 0.5, linewidth = 1, linestyle = '--')
ax_1.set_xlabel(r'$c_t$')
ax_1.set_ylabel(r'$p^{\sigma^2}(o_t|c_t=-0.01)$')
ax_4.bar([0,1], val,color='k')
ax_4.set_ylim([0, 1])
ax_4.set_ylabel(r'$p(d_t|c_t=-0.01)$')
ax_4.set_xlabel(r'$d_t$')

# Example 3: Zero contrast difference
mu = 0
sigma = 0.02
p_o_giv_u = stats.norm.pdf(U, mu, sigma)
val = [stats.norm.cdf(0, mu, sigma), 1 - stats.norm.cdf(0, mu, sigma)]#0.5
ax_2.plot(U, p_o_giv_u, color='k')
ax_2.axvline(x=0, color='k', alpha = 0.5, linewidth = 1, linestyle = '--')
ax_2.set_xlabel(r'$c_t$')
ax_2.set_ylabel(r'$p^{\sigma^2}(o_t|c_t=0)$')
ax_5.bar([0,1], val, color='k')
ax_5.set_ylim([0, 1])
ax_5.set_xlabel(r'$d_t$')
ax_5.set_ylabel(r'$p(d_t|c_t=0)$')
ax_5.set_xticks([0, 1])
sns.despine()

# Adjust plot properties
plt.subplots_adjust(hspace=0.4, wspace=0.5)

# Label letters
texts = ['a', 'b', 'c', 'd', 'e', 'f']

# Add labels
label_subplots(f, texts)

# Save figure
plt.savefig('gb_figures/gb_sm_figure_7.pdf', dpi=400, transparent=True)

# Show plot
plt.show()