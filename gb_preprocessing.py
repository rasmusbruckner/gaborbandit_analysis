""" This script runs the preprocessing steps

    1. Load BIDS formatted data
    2. Create preprocessed data frames for each experiment
    3. Save data

"""


import sys
import os
from fnmatch import fnmatch
from gb_loaddata import loaddata
from gb_compute_misses import compute_misses

# ---------------------------
# 1. Load BIDS-formatted data
# ---------------------------

# Set path and directory
my_path = '/Users/rasmus/Dropbox/gabor_bandit/code/gaborbandit_analysis/gb_data'
os.chdir(my_path)

filenames = []
for path, subdirs, files in os.walk(my_path + '/BIDS/behav/'):
    for name in files:
        if fnmatch(name, "*.tsv"):
            filenames.append(os.path.join(path, name))

# Put pseudonomized BIDS data in a single data frame
data_pn = loaddata(filenames)

# Number of subjects
N = len(list(set(data_pn['participant'])))

# ------------------------------------------------------
# 2. Create preprocessed data frames for each experiment
# ------------------------------------------------------

# Experiment 1 - Perceptual decision making
exp1_data = data_pn[(data_pn['d_t'].notna()) & (data_pn['trial_type'] == 'patches') & (data_pn['complete'])].copy()

# Compute number of missed trials
n_misses_1 = 100 - exp1_data['id'].value_counts()
n_misses_min_1, n_misses_max_1, n_misses_mean_1, n_misses_sem_1 = compute_misses(n_misses_1, N)

# Experiment 1 - Perceptual decision making
exp1_data_recov = data_pn[(data_pn['trial_type'] == 'patches') & (data_pn['complete'])].copy()
exp1_data_recov.loc[:, 'b_t'] = 0

# Experiment 2 - Economic decisions making without perceptual uncertainty
exp2_raw = data_pn[(data_pn['trial_type'] == 'main_safe')].copy()  # raw data to compute misses
exp2_data = data_pn[(data_pn['missIndex'] == 0) & (data_pn['decision2.corr'].notna()) &
                    (data_pn['trial_type'] == 'main_safe')].copy()  # preprocessed data for analysis

# Extract number of blocks
exp2_nb = len(set(list(exp2_data['b_t'])))

# Compute number of missed trials
n_misses_2 = exp2_raw['id'].value_counts() - 150
n_misses_min_2, n_misses_max_2, n_misses_mean_2, n_misses_sem_2 = compute_misses(n_misses_2, N)

# Experiment 3 - Economic decision making with perceptual uncertainty
exp3_raw = data_pn[(data_pn['trial_type'] == 'main_unc')].copy()  # raw data to compute misses
exp3_data = data_pn[(data_pn['missIndex'] == 0) & (data_pn['decision2.corr'].notna()) &
                    (data_pn['trial_type'] == 'main_unc')].copy()  # preprocessed data for analysis

# Compute number of missed trials
n_misses_3 = exp3_raw['id'].value_counts() - 300
n_misses_min_3, n_misses_max_3, n_misses_mean_3, n_misses_sem_3 = compute_misses(n_misses_3, N)

# Extract number of blocks
exp3_nb = len(set(list(exp3_data['blockNumber'])))

# Check if we have the same number of participants for every experiment
# ---------------------------------------------------------------------
n_subj_1 = len(list(set(exp1_data['participant'])))
n_subj_2 = len(list(set(exp2_data['participant'])))
n_subj_3 = len(list(set(exp3_data['participant'])))

if not n_subj_1 == n_subj_2 == n_subj_3:
    sys.exit("Unequal number of participants")

# -------------
#  3. Save data
# -------------

exp1_data.to_pickle('gb_exp1_data.pkl')
exp1_data_recov.to_pickle('gb_exp1_data_recov.pkl')
exp2_data.to_pickle('gb_exp2_data.pkl')
exp3_data.to_pickle('gb_exp3_data.pkl')
