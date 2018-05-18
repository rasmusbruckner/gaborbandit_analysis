""" This script runs the parameter estimation and the evaluation of the agent-based computational models

    1. Estimate sensory noise parameter sigma based on Experiment 1
    2. Estimate sensory noise parameter sigma based on Experiment 3
    3. Estimate inverse temperature parameter beta based on Experiment 2
    4. Estimate lambda parameter based on Experiment 3
    5. Evaluate agent A0 - A3 based on Experiment 3

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GbTaskVars import TaskVars
from GbEstVars import EstVars
from gb_parallelest import parallelest
from GbEstimation import GbEstimation
from gb_eval import evalfun

# Set random number generator for reproducible results
np.random.seed(123)

# Load preprocessed data of all experiments
exp1_data = pd.read_pickle('gb_data/gb_exp1_data.pkl')
exp2_data = pd.read_pickle('gb_data/gb_exp2_data.pkl')
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Number of subjects
N = len(list(set(exp1_data['participant'])))

# Task parameters
task_vars = TaskVars()
task_vars.T = 100

# Estimation parameters
est_vars = EstVars(task_vars)
est_vars.n_sim = N

# Estimation object
gb_estimation = GbEstimation(est_vars)

# Initialize data frame for estimated parameters
params = pd.DataFrame(index=range(0, N), dtype='float')

# Set number of starting points and kernels for parallel estimation
est_vars.n_sp = 1
est_vars.n_ker = 4

# 1. Estimate sensory noise parameter sigma based on Experiment 1
# ---------------------------------------------------------------

# Adjust est_vars
est_vars.experiment = 1
est_vars.agent = 1

# Estimate sigma parameter
results_df = parallelest(exp1_data, est_vars)

# Add results to data frame
params['id'] = results_df['id']
params['sigma'] = results_df['minimum']
params['exp1_llh'] = results_df['llh']
params['exp1_d_BIC'] = results_df['d_BIC']
plt.bar(np.arange(N), results_df['minimum'])

# 2. Estimate sensory noise parameter sigma based on Experiment 3
# ---------------------------------------------------------------

# Adjust est_vars
est_vars.experiment = 1

# Estimate sigma parameter
results_df = parallelest(exp3_data, est_vars)

# Add results to data frame
params['id'] = results_df['id']
params['sigma_exp3'] = results_df['minimum']
plt.bar(np.arange(N), results_df['minimum'])

# 3. Estimate inverse temperature parameter beta based on Experiment 2
# --------------------------------------------------------------------

# Set number of blocks for estimation
est_vars.B = len(set(list(exp2_data['b_t'])))
est_vars.T = sum(exp2_data[exp2_data['id'] == 0]['b_t'] == 0)

# Indicate that we're fitting economic decisions and evaluating perceptual decisions
est_vars.type = 1

# Use agent A1
est_vars.agent = 1

est_vars.experiment = 2
fixed_params = pd.DataFrame(index=range(0, N), dtype='float')
fixed_params['sigma'] = params['sigma']

# todo: klären, welches wir nehmen. Wenn exp3, dann über numerische Lösung nachdenken
fixed_params['sigma'] = params['sigma_exp3']
fixed_params[fixed_params['sigma'] < 0.015] = 0.015

# Estimate beta parameter
results_df = parallelest(exp2_data, est_vars, fixed_params=fixed_params)

plt.figure()
plt.bar(np.arange(N), results_df['minimum'])

# Add results to data frame
params['beta'] = results_df['minimum']
params['exp2_llh'] = results_df['llh']
params['exp2_d_BIC'] = results_df['d_BIC']
params['exp2_a_BIC'] = results_df['a_BIC']


# 4. Estimate lambda parameter based on Experiment 3
# --------------------------------------------------

# Evaluate mixture model
est_vars.agent = 3  # models versions that are simulated
est_vars.experiment = 3

# Extract number of blocks
est_vars.B = len(set(list(exp3_data['b_t'])))

# Add estimated beta parameter to the set of fixed parameters
fixed_params['beta'] = params['beta']

# todo: klären, welches wir nehmen. Wenn exp3, dann über numerische Lösung nachdenken
fixed_params['sigma'] = params['sigma_exp3']
fixed_params[fixed_params['sigma'] < 0.015] = 0.015

# Estimate mixture parameter lambda
results_df = parallelest(exp3_data, est_vars, fixed_params=fixed_params)

# Add results to data frame
params['lambda'] = results_df['minimum']
params['A3_llh'] = results_df['llh']
params['A3_d_BIC'] = results_df['d_BIC']
params['A3_a_BIC'] = results_df['a_BIC']

plt.figure()
plt.bar(np.arange(N), results_df['minimum'])

# 5. Evaluate agent A0 - A2 based on Experiment 3
# -----------------------------------------------
df_bic = evalfun(exp3_data, est_vars, fixed_params.copy())
params['A0_llh'] = df_bic.loc[df_bic['agent'] == 0, 'llh']
params['A1_llh'] = df_bic.loc[df_bic['agent'] == 1, 'llh']
params['A2_llh'] = df_bic.loc[df_bic['agent'] == 2, 'llh']
params['A0_d_BIC'] = df_bic.loc[df_bic['agent'] == 0, 'd_BIC']
params['A1_d_BIC'] = df_bic.loc[df_bic['agent'] == 1, 'd_BIC']
params['A2_d_BIC'] = df_bic.loc[df_bic['agent'] == 2, 'd_BIC']
params['A0_a_BIC'] = df_bic.loc[df_bic['agent'] == 0, 'a_BIC']
params['A1_a_BIC'] = df_bic.loc[df_bic['agent'] == 1, 'a_BIC']
params['A2_a_BIC'] = df_bic.loc[df_bic['agent'] == 2, 'a_BIC']

# Save parameters
params.to_pickle('gb_data/modelbased.pkl')

# Save BIC as csv for Bayesian model comparison
# ---------------------------------------------
part_bic_mat = np.full([N, 4], np.nan)
part_bic_mat[:, 0] = params['A0_d_BIC'] + params['A0_a_BIC']
part_bic_mat[:, 1] = params['A1_d_BIC'] + params['A2_a_BIC']
part_bic_mat[:, 2] = params['A2_d_BIC'] + params['A2_a_BIC']
part_bic_mat[:, 3] = params['A3_d_BIC'] + params['A3_a_BIC']
np.savetxt('gb_data/part_bic_mat.csv', part_bic_mat, delimiter=',')

# Plot estimates
plt.show()
