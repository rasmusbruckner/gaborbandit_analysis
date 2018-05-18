""" This script runs the preprocessing steps

    1) Load pseudonymized data
    2) Create preprocessed data frames for each experiment
    3) Save data

"""
import numpy as np
import pandas as pd
import os
import sys

# Add path
#path = '/Users/rasmus/Dropbox/gabor_bandit/code/python'
#os.chdir(path)

# 1) Load pseudonymized data
# --------------------------

data_pn = pd.read_pickle('gb_data/gb_data_pn.pkl')

# Number of subjects
N = len(list(set(data_pn['participant'])))

# 2) Create preprocessed data frames for each experiment
# ------------------------------------------------------

# Experiment 1 - Perceptual decision making
exp1_data = data_pn[(data_pn['d_t'].notna()) & (data_pn['whichLoop'] == 'patches') & (data_pn['complete'])].copy()

# Compute number of missed trials
T__miss_exp1 = 100 - exp1_data['id'].value_counts()
T_exp1_min = np.min(np.min(T__miss_exp1))
T_exp1_max = np.max(T__miss_exp1)
T_exp1_mean = np.mean(T__miss_exp1)
T_exp1_sd = np.std(T__miss_exp1)
T_exp1_sem = T_exp1_sd/np.sqrt(N)

# Experiment 1 - Perceptual decision making
exp1_data_recov = data_pn[(data_pn['whichLoop'] == 'patches') & (data_pn['complete'])].copy()
exp1_data_recov.loc[:, 'b_t'] = 0

# Experiment 2 - Economic decisions making without perceptual uncertainty
exp2_data = data_pn[(data_pn['missIndex'] == 0) & (data_pn['decision2.corr'].notna()) &
                    (data_pn['whichLoop'] == 'main_safe')].copy()
exp2_raw = data_pn[(data_pn['whichLoop'] == 'main_safe')].copy()

# Extract number of blocks
exp2_nb = len(set(list(exp2_data['b_t'])))

# Compute number of missed trials
T__miss_exp2 = exp2_raw['id'].value_counts() - 150
T_exp2_min = np.min(np.min(T__miss_exp2))
T_exp2_max = np.max(T__miss_exp2)
T_exp2_mean = np.mean(T__miss_exp2)
T_exp2_sd = np.std(T__miss_exp2)
T_exp2_sem = T_exp2_sd/np.sqrt(N)

# Experiment 3 - Economic decision making with perceptual uncertainty
exp3_data = data_pn[(data_pn['missIndex'] == 0) & (data_pn['decision2.corr'].notna()) &
                    (data_pn['whichLoop'] == 'main_unc')].copy()

# Compute number of missed trials
exp3_raw = data_pn[(data_pn['whichLoop'] == 'main_unc')].copy()
T__miss_exp3 = exp3_raw['id'].value_counts() - 300
T_exp3_min = np.min(np.min(T__miss_exp3))
T_exp3_max = np.max(T__miss_exp3)
T_exp3_mean = np.mean(T__miss_exp3)
T_exp3_sd = np.std(T__miss_exp3)
T_exp3_sem = T_exp3_sd/np.sqrt(N)

# Extract number of blocks
exp3_nb = len(set(list(exp3_data['blockNumber'])))

# Check if we have the same number of participants for every experiment
# ---------------------------------------------------------------------
n_subj_1 = len(list(set(exp1_data['participant'])))
n_subj_2 = len(list(set(exp2_data['participant'])))
n_subj_3 = len(list(set(exp3_data['participant'])))

if not n_subj_1 == n_subj_2 == n_subj_3:
    sys.exit("Unequal number of participants")

#  3) Save data
# -------------
exp1_data.to_pickle('gb_data/gb_exp1_data.pkl')
exp1_data_recov.to_pickle('gb_data/gb_exp1_data_recov.pkl')
exp2_data.to_pickle('gb_data/gb_exp2_data.pkl')
exp3_data.to_pickle('gb_data/gb_exp3_data.pkl')
