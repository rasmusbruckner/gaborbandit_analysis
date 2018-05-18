""" This scripts implements the descriptive analysis

    1. t-tests for perceptual choice performance in Experiment 1 and 3
    2. t-test between economic choice performance in Experiment 2 and 3.
    3. Generation of SM Figure 1
    4. Generation of Figure 2

"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from GbAgentVars import AgentVars
from GbAgent import Agent
from scipy import stats
import pickle

# Use Latex for plotting
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

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


# Adjust general figure properties
blue2 = (0.3050288350634371, 0.3050287157042084, 0.42438794443608646)  # color for Experiment 1
blue1 = (0.44042675893886962, 0.48579509331068693, 0.56542666115994655)  # color for Experiment 2
blue3 = (0.48475201845444055, 0.5467423147558812, 0.6097519306486952)  # color for Experiment 3

# Load preprocessed data of all experiments and model-based results
exp1_data = pd.read_pickle('gb_data/gb_exp1_data.pkl')
exp2_data = pd.read_pickle('gb_data/gb_exp2_data.pkl')
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')
sub_params = pd.read_pickle('gb_data/modelbased.pkl')

# Get all IDs
all_id = list(set(exp1_data['participant']))

# Number of participants and blocks
n_subj = len(all_id)
n_blocks2 = len(list(set(exp2_data['blockNumber'])))
n_blocks3 = len(list(set(exp3_data['blockNumber'])))

# Compute performances
# --------------------

# Add trial number to data frames
exp2_data.loc[:, 'trial'] = np.tile(np.linspace(0, 24, 25), [len(all_id)*n_blocks2])
exp3_data.loc[:, 'trial'] = np.tile(np.linspace(0, 24, 25), [len(all_id)*n_blocks3])

# Compute p(d_t = 1), i.e., right perceptual decisions based on sigma parameter
# from the model-based analyses
agent_vars = AgentVars()  # agent variables
agent = Agent(agent_vars)  # initialize agent object
U = np.linspace(-0.1, 0.1, 100)  # range of observations

# Initialize arrays for Experiment 1 and 3
exp1_pi1 = np.full([len(all_id), len(U)], np.nan)
exp3_pi1 = np.full([len(all_id), len(U)], np.nan)

# Cycle over participants
for i in range(0, n_subj):

    agent.sigma = sub_params['sigma'][i]  # set current sigma paramter for Exp. 1
    _, exp1_pi1[i, :] = agent.p_s_giv_o(U)

    agent.sigma = sub_params['sigma_exp3'][i]  # set current sigma paramter for Exp. 3
    _, exp3_pi1[i, :] = agent.p_s_giv_o(U)

# Experiment 1
# ------------
# Mean perceptual decision making performance
exp1_perc_part = exp1_data.groupby(['participant'])['decision1.corr'].mean()
exp1_perc_mean = np.mean(exp1_perc_part)
exp1_perc_sd = np.std(exp1_perc_part)
exp1_perc_sem = exp1_perc_sd/np.sqrt(n_subj)

# p(d_t = 1|c_t)
exp1_pi1_mean = np.mean(exp1_pi1, 0)
exp1_pi1_sd = np.std(exp1_pi1_mean, 0)
exp1_pi1_sem = exp1_pi1_sd/np.sqrt(len(all_id))

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

# p(d_t = 1|c_t)
exp3_pi1_mean = np.mean(exp3_pi1, 0)
exp3_pi1_sd = np.std(exp3_pi1_mean, 0)
exp3_pi1_sem = exp3_pi1_sd/np.sqrt(len(all_id))

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

# Put variables for bar plots together
perc_mean = [exp3_perc_mean, exp2_perc_mean, exp1_perc_mean]
perc_sd = [exp3_perc_sd, exp2_perc_sd, exp1_perc_sd]
perc_sem = [exp3_perc_sem, exp2_perc_sem, exp1_perc_sem]
econ_mean = [exp3_econ_mean, exp2_econ_mean]
econ_sd = [exp3_econ_sd, exp2_econ_sd]
econ_sem = [exp3_econ_sem, exp2_econ_sem]

# Save trial-wise performance of Exp 3 for Figure 4 (model-based results)
df_econ_trial = pd.DataFrame()
df_econ_trial['trial_mean'] = exp3_trial_mean
df_econ_trial['trial_sem'] = exp3_trial_sem
df_econ_trial['trial_sd'] = exp3_trial_sd
f = open('gb_data/trial_wise_part.pkl', 'wb')
pickle.dump(df_econ_trial, f)
f.close()

# 1. t-tests for perceptual decision making
# -----------------------------------------
t, p = stats.ttest_rel(exp1_perc_part, exp3_perc_part)
print('Perceptual decision making:\nt=%s, p=%s ' % (t, p))

# 2. t-tests for economic decision making
# -----------------------------------------
t, p = stats.ttest_rel(exp2_econ_part, exp3_econ_part)
print('Economic decision making:\nt=%s, p=%s ' % (t, p))


# 3. SM Figure 1
# --------------

# Prepare figure
fig = plt.figure(0, (8, 8))
ax_1 = plt.subplot2grid((3, 2), (0, 0))
ax_2 = plt.subplot2grid((3, 2), (1, 0))
ax_3 = plt.subplot2grid((3, 2), (2, 0))
ax_4 = plt.subplot2grid((3, 2), (1, 1))
ax_5 = plt.subplot2grid((3, 2), (2, 1))

# Experiment 1
# ------------
# Mean perceptual decision making performance
ax_1.bar(np.arange(len(all_id)), exp3_perc_part, color=blue1)
ax_1.axhline(exp1_perc_mean, color='black', lw=0.5, linestyle='--')
ax_1.set_ylim([0.5, 1])
ax_1.set_title('PDM')
ax_1.set_xlabel('Participant')
ax_1.set_ylabel('Perceptual Choice\nPerformance')

# Experiment 2
# ------------
# Mean perceptual decision making performance
ax_2.bar(np.arange(len(all_id)), exp2_perc_part, color=blue1)
ax_2.axhline(exp2_perc_mean, color='black', lw=0.5, linestyle='--')
ax_2.set_ylim([0.5, 1])
ax_2.set_title('EDM')
ax_2.set_xlabel('Participant')
ax_2.set_ylabel('Perceptual Choice\nPerformance')

# Mean economic decision making performance
ax_4.bar(np.arange(len(all_id)), exp2_econ_part, color=blue1)
ax_4.axhline(exp2_econ_mean, color='black', lw=0.5, linestyle='--')
ax_4.set_ylim([0.5, 1])
ax_4.set_title('EDM')
ax_4.set_ylabel('Economic Choice\nPerformance')
ax_4.set_xlabel('Participant')

# Experiment 3
# ------------
# Mean perceptual decision making performance
ax_3.bar(np.arange(len(all_id)), exp3_perc_part, color=blue1)
ax_3.axhline(exp3_perc_mean, color='black', lw=0.5, linestyle='--')
ax_3.set_ylim([0.5, 1])
ax_3.set_title('GB')
ax_3.set_xlabel('Participant')
ax_3.set_ylabel('Perceptual Choice\nPerformance')

# Mean economic decision making performance
ax_5.bar(np.arange(len(all_id)), exp3_econ_part, color=blue1)
ax_5.axhline(exp3_econ_mean, color='black', lw=0.5, linestyle='--')
ax_5.set_ylim([0.5, 1])
ax_5.set_title('GB')
ax_5.set_ylabel('Economic Choice\nPerformance')
ax_5.set_xlabel('Participant')

# Adjust figure space
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.10, right=0.9, hspace=0.5, wspace=0.35)
sns.despine()

# Add figure lables
fig.text(0.04, 0.95, "A)", horizontalalignment='left', verticalalignment='center')
fig.text(0.04, 0.61, "B)", horizontalalignment='left', verticalalignment='center')
fig.text(0.5, 0.61, "C)", horizontalalignment='left', verticalalignment='center')
fig.text(0.04, 0.28, "D)", horizontalalignment='left', verticalalignment='center')
fig.text(0.5, 0.28, "E)", horizontalalignment='left', verticalalignment='center')


# Save figure
# -----------
savename = 'gb_figures/SM_Fig1.pdf'
plt.savefig(savename)

# 4. Figure 2
# -----------

# Prepare figure

fig = plt.figure(5, figsize=(8, 6))
ax_0 = plt.subplot2grid((2, 3), (0, 0))
ax_1 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
ax_2 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
ax_3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)

# Adjust figure space
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.10, right=0.9, hspace=0.25, wspace=0.35)

# Mean perceptual decision making performance
# -------------------------------------------
barlist = ax_0.bar([1, 2, 3], perc_mean, yerr=perc_sem, alpha=1, edgecolor='k',
                   error_kw=dict(ecolor='k', lw=1, capsize=5, capthick=1))
barlist[0].set_facecolor(blue1)
barlist[1].set_facecolor(blue1)
barlist[2].set_facecolor(blue1)
ax_0.set_ylim([0.5, 1])
ax_0.set_xlabel('Experiment')
ax_0.set_ylabel('Perceptual Choice Performance')
ax_0.set_xticks([1, 2, 3])
ax_0.set_xticklabels(['GB', 'EDM', 'PDM'])

# Mean economic decision making performance
# -----------------------------------------
barlist = ax_2.bar([2, 3], econ_mean, yerr=econ_sem, alpha=1, edgecolor='k',
                   error_kw=dict(ecolor='k', lw=1, capsize=5, capthick=1))
barlist[0].set_facecolor(blue1)
barlist[1].set_facecolor(blue1)
ax_2.set_ylim([0.5, 1])
ax_2.set_xticks(np.arange(2, 4, step=1))
ax_2.set_xlabel('Experiment')
ax_2.set_ylabel('Economic Choice Performance')
ax_2.set_xticks([2, 3])
ax_2.set_xticklabels(['GB', 'EDM'])

# p(d_t = 1|c_t)
# -------------
# Experiment 1
ax_1.plot(np.arange(100), exp1_pi1_mean, color='k', linestyle='--')
ax_1.fill_between(np.arange(100), exp1_pi1_mean-exp1_pi1_sem, exp1_pi1_mean+exp1_pi1_sem,
                  edgecolor='k', facecolor=blue1, alpha=0.5)

# Experiment 3
ax_1.plot(np.arange(100), exp3_pi1_mean, color='k', linestyle='-')
ax_1.fill_between(np.arange(100), exp3_pi1_mean-exp3_pi1_sem, exp3_pi1_mean+exp3_pi1_sem,
                  edgecolor='k', facecolor=blue1, alpha=0.5)

# Adjust plot properties
ax_1.set_ylabel(r'Psychometric Function $(\pi_1)$')
ax_1.set_xlabel(r'Contrast Difference ($c_t$)')
a = ax_1.get_xticks().tolist()
a[1] = '-0.1'
a[2] = '-0.06'
a[3] = '-0.02'
a[4] = '0.02'
a[5] = '0.06'
a[6] = '0.1'
ax_1.set_xticklabels(a)
ax_1.legend(["PDM", "GB"], loc=4)

# Economic decision making performance across trials
# --------------------------------------------------

x = np.linspace(1,25, 25)

# Experiment 2
ax_3.plot(x, exp2_trial_mean, color='k', linestyle='--')
ax_3.fill_between(x, exp2_trial_mean-exp2_trial_sem, exp2_trial_mean+exp2_trial_sem,
                  edgecolor='k', facecolor=blue1, alpha=0.5)

# Experiment 3
ax_3.plot(x, exp3_trial_mean, color='k', linestyle='-')
ax_3.fill_between(x, exp3_trial_mean-exp3_trial_sem, exp3_trial_mean+exp3_trial_sem,
                  edgecolor='k', facecolor=blue1, alpha=0.5)

# Adjust plot properties
ax_3.set_xlabel(r'Trial ($t$)')
ax_3.set_ylim([0.4, 1])
ax_3.legend(["EDM", "GB"], loc=4)
ax_3.set_ylabel('Economic Choice Performance')

# Use figure space more efficiently
sns.despine()

# Add figure labels
fig.text(0.02, 0.82, "A)", horizontalalignment='left', verticalalignment='center')
fig.text(0.33, 0.82, "B)", horizontalalignment='left', verticalalignment='center')
fig.text(0.02, 0.49, "C)", horizontalalignment='left', verticalalignment='center')
fig.text(0.33, 0.49, "D)", horizontalalignment='left', verticalalignment='center')

# Save plot
savename = 'gb_figures/Fig2.pdf'
plt.savefig(savename)

plt.show()
