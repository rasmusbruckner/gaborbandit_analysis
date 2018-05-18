""" This script implements the model and parameter recovery studies

    1. Run model recovery study
    2. Run parameter recovery study
"""

import numpy as np
import pandas as pd
from GbTaskVars import TaskVars
from GbSimVars import SimVars
from GbEstVars import EstVars
from gb_parallelest import parallelest
from gb_recovsim import recovsim
from gb_eval import evalfun


# Set random number generator for reproducible results
np.random.seed(123)

# Load preprocessed data of all experiments
exp1_data = pd.read_pickle('gb_data/gb_exp1_data_recov.pkl')
exp2_data = pd.read_pickle('gb_data/gb_exp2_data.pkl')
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Load estimated parameter of participants
sub_params = pd.read_pickle('gb_data/modelbased.pkl')

# Number of subjects
N = len(list(set(exp1_data['participant'])))

# Number of true parameters to be recovered
n_trparams = 6

# Number of simulations
n_sim = N * n_trparams

# Indicate index of true parameter
which_param = np.repeat(np.arange(n_trparams), N)

# Here we use the sigma and beta parameters of our participants
fixed_params = pd.DataFrame(index=range(0, n_sim), dtype='float')
fixed_params['sigma'] = pd.concat([sub_params['sigma']]*n_trparams, ignore_index=True)
fixed_params['beta'] = pd.concat([sub_params['beta']]*n_trparams, ignore_index=True)

# Task parameters
task_vars = TaskVars()

# Simulation parameters
sim_vars = SimVars()
sim_vars.take_pd = 0  # here we don't take perceptual decisions of participants
sim_vars.N = N
sim_vars.n_sim = n_sim

# Estimation parameters
est_vars = EstVars(task_vars)
est_vars.d_bnds = [(0.01, 0.1), ]
est_vars.a_bnds = [(0, 20), ]
est_vars.l_bnds = [(0, 1), ]
est_vars.N = N
est_vars.n_ker = 4

# Initialize data frames
recov_params = pd.DataFrame(index=range(0, n_sim), dtype='float')  # recovered parameters
model_recov = pd.DataFrame(index=range(0, N), dtype='float')  # recovered models

# Add index of true parameter to data frame
recov_params.loc[:, 'which_param'] = which_param


# 1. Run model recovery study
# ---------------------------

# Adjust est_vars
n_sim = N
sim_vars.n_sim = n_sim
est_vars.n_sim = n_sim

# Indicate that mixture model should be evaluated
eval_a3 = 1

# Adjust sim_vars
sim_vars.param_range = [0.5]
sim_vars.take_pd = 0  # here we don't take perceptual decisions of participants


# Select A0
# ---------
agent = 0

# Adjust sim_vars
sim_vars.agent = agent

# Generate data for model recovery
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=sub_params)

# Evaluate all models

df_bic_0 = evalfun(df_subj, est_vars, sub_params, eval_a3)
df_bic_0.to_pickle('gb_data/df_bic_0.pkl')


# Select A1
# ---------
agent = 1

# Adjust sim_vars
sim_vars.agent = agent

# Generate data for model recovery
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=sub_params)

# Evaluate all models
df_bic_1 = evalfun(df_subj, est_vars, sub_params, eval_a3)
df_bic_1.to_pickle('gb_data/df_bic_1.pkl')

# Select A2
# ---------
agent = 2

# Adjust sim_vars
sim_vars.agent = agent

# Generate data for model recovery
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=sub_params)

# Evaluate all models
df_bic_2 = evalfun(df_subj, est_vars, sub_params, eval_a3)
df_bic_2.to_pickle('gb_data/df_bic_2.pkl')

# Select A3
# ---------
agent = 3

# Adjust sim_vars
sim_vars.agent = agent

# Generate data for model recovery
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=sub_params)

# Evaluate all models
df_bic_3 = evalfun(df_subj, est_vars, sub_params, eval_a3)
df_bic_3.to_pickle('gb_data/df_bic_3.pkl')


# Save BIC as csv for Bayesian model comparison
# ---------------------------------------------
recov_bic_mat = np.full([N, 16], np.nan)
recov_bic_mat[:, 0] = df_bic_0[df_bic_0['agent'] == 0]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 0]['a_BIC']
recov_bic_mat[:, 1] = df_bic_0[df_bic_0['agent'] == 1]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 1]['a_BIC']
recov_bic_mat[:, 2] = df_bic_0[df_bic_0['agent'] == 2]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 2]['a_BIC']
recov_bic_mat[:, 3] = df_bic_0[df_bic_0['agent'] == 3]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 3]['a_BIC']

recov_bic_mat[:, 4] = df_bic_1[df_bic_1['agent'] == 0]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 0]['a_BIC']
recov_bic_mat[:, 5] = df_bic_1[df_bic_1['agent'] == 1]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 1]['a_BIC']
recov_bic_mat[:, 6] = df_bic_1[df_bic_1['agent'] == 2]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 2]['a_BIC']
recov_bic_mat[:, 7] = df_bic_1[df_bic_1['agent'] == 3]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 3]['a_BIC']

recov_bic_mat[:, 8] = df_bic_2[df_bic_2['agent'] == 0]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 0]['a_BIC']
recov_bic_mat[:, 9] = df_bic_2[df_bic_2['agent'] == 1]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 1]['a_BIC']
recov_bic_mat[:, 10] = df_bic_2[df_bic_2['agent'] == 2]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 2]['a_BIC']
recov_bic_mat[:, 11] = df_bic_2[df_bic_2['agent'] == 3]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 3]['a_BIC']

recov_bic_mat[:, 12] = df_bic_3[df_bic_3['agent'] == 0]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 0]['a_BIC']
recov_bic_mat[:, 13] = df_bic_3[df_bic_3['agent'] == 1]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 1]['a_BIC']
recov_bic_mat[:, 14] = df_bic_3[df_bic_3['agent'] == 2]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 2]['a_BIC']
recov_bic_mat[:, 15] = df_bic_3[df_bic_3['agent'] == 3]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 3]['a_BIC']

np.savetxt('gb_data/recov_bic_mat.csv', recov_bic_mat, delimiter=',')


# 2. Run parameter recovery study
# -------------------------------

# Update est_vars
est_vars.n_sim = N*n_trparams

# Sigma recovery
# --------------

# Select A1
agent = 1

# Adjust task_vars
task_vars.T = 100
task_vars.B = 1
task_vars.experiment = 1

# Range of sigma values
sigma_range = np.linspace(0.01, 0.09, n_trparams)

# Adjust sim_vars
sim_vars.agent = agent
sim_vars.param_range = sigma_range

# Generate data for parameter recovery
df_subj = recovsim(task_vars, sim_vars, exp1_data)

# Adjust est_vars
est_vars.agent = agent
est_vars.n_blocks = 1
est_vars.experiment = task_vars.experiment
est_vars.T = task_vars.T

# Estimate sigma parameter
results_df = parallelest(df_subj, est_vars)
recov_params.loc[:, 'sigma'] = results_df['minimum'].copy()
recov_params.loc[:, 'sigma_bias'] = results_df['minimum'] - np.repeat(sigma_range, N)


# Recovery for beta
# -----------------

# Adjust task_vars
task_vars.T = 25
task_vars.B = 6
task_vars.experiment = 2

# Range of beta values
beta_range = np.linspace(0, 18, n_trparams)

# Adjust sim_vars
sim_vars.param_range = beta_range

# Generate data for parameter recovery
df_subj = recovsim(task_vars, sim_vars, exp2_data, opt_params=sub_params)

# Adjust est_vars
est_vars.B = 6
est_vars.experiment = task_vars.experiment
est_vars.T = task_vars.T

# Estimate beta parameter
results_df = parallelest(df_subj, est_vars, fixed_params=fixed_params)
recov_params.loc[:, 'beta'] = results_df['minimum'].copy()
recov_params.loc[:, 'beta_bias'] = results_df['minimum'] - np.repeat(beta_range, N)


# Lambda recovery
# ----------------

# Select A3
agent = 3

# Adjust task_vars
task_vars.B = 12
task_vars.experiment = 3

# Range of lambda values
lambda_range = np.linspace(0, 1, n_trparams)

# Adjust sim_vars
sim_vars.param_range = lambda_range
sim_vars.agent = agent

# Generate data for parameter recovery
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=sub_params)

# Adjust est_vars
est_vars.agent = agent
est_vars.B = 12
est_vars.experiment = task_vars.experiment

# # Estimate lambda parameter
results_df = parallelest(df_subj, est_vars, fixed_params=fixed_params)
recov_params.loc[:, 'lambda'] = results_df['minimum'].copy()
recov_params.loc[:, 'lambda_bias'] = results_df['minimum'] - np.repeat(lambda_range, N)

# Save parameter recovery results
recov_params.to_pickle('gb_data/param_recov.pkl')
