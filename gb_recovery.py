""" This script implements the model and parameter recovery studies

    1. Load data and set recovery parameters
    2. Run model recovery study
    3. Run parameter recovery study
"""

import numpy as np
import pandas as pd
from GbTaskVars import TaskVars
from GbSimVars import SimVars
from GbEstVars import EstVars
from gb_parallelest import parallelest
from gb_recovsim import recovsim
from gb_eval import evalfun


# todo: take_pd nochmal in code checken, eventuell wegmachen, wo soll man das für gebrauchen?
# model evaluation basierend auf LL und nicht BIC sonst geht pEP nicht -- aber schauen, dass free Params = 0!
# überprüfen, ob ll genau gleich ist!
# überprüfen, ob model_params nicht durch eval_params geändert wird. habe jetzt copy() eingebaut, muss
# aber verifiziert werden

# Set random number generator for reproducible results
np.random.seed(123)

# ----------------------------------------
# 1. Load data and set recovery parameters
# ----------------------------------------

# Load preprocessed data
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Create data frame with sigma, beta, lambda and alpha values that are used in model and parameter recovery
sim_params = pd.DataFrame()
sim_params['sigma'] = np.repeat([0.015, 0.025, 0.035, 0.045], 64)
sim_params['beta'] = np.tile(np.repeat([4, 8, 12, 16], 16), 4)
sim_params['lambda'] = np.tile(np.repeat([0.2, 0.4, 0.6, 0.8], 4), 16)
sim_params['alpha'] = np.tile([0.1, 0.3, 0.5, 0.7], 64)

# Create data frame with parameters for each model
model_params = pd.DataFrame()
model_params['sigma'] = sim_params['sigma']
model_params['beta_A1'] = model_params['beta_A2'] = model_params['beta_A3'] = model_params['beta_A4'] \
    = model_params['beta_A5'] = model_params['beta_A6'] = sim_params['beta']
model_params['lambda_A3'] = model_params['lambda_A6'] = sim_params['lambda']
model_params['alpha_A4'] = model_params['alpha_A5'] = model_params['alpha_A6'] = sim_params['alpha']

# Create data frame used for evaluation of models in model recovery, where lambda is fixed to 0.5
eval_params = model_params.copy()
eval_params['lambda_A3'] = eval_params['lambda_A6'] = 0.5

# Number of subjects
N = len(list(set(exp3_data['participant'])))

# ---------------------------
# 2. Run model recovery study
# ---------------------------

# Task parameters
task_vars = TaskVars()

# Simulation parameters
sim_vars = SimVars()
sim_vars.take_pd = 0  # here we don't take perceptual decisions of participants
sim_vars.N = N

# Estimation parameters
est_vars = EstVars(task_vars)

# todo: was kommt hiervon in parameter recovery?
# est_vars.n_ker = 4
# est_vars.n_sp = 10
# est_vars.rand_sp = True
est_vars.real_data = False  # todo: das nochmal kontrollieren!

# Agent A0
# ---------
sim_vars.agent = 0
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_0 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_0.to_pickle('gb_data/df_bic_0.pkl')
# df_bic_0.to_pickle('gb_data/df_bic_0_new.pkl')
df_bic_0.to_pickle('gb_data/df_bic_0_final.pkl')  # save data

# Agent A1
# ---------
sim_vars.agent = 1
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_1 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_1.to_pickle('gb_data/df_bic_1.pkl')
# df_bic_1.to_pickle('gb_data/df_bic_1_new.pkl')
df_bic_1.to_pickle('gb_data/df_bic_1_final.pkl')  # save data

# Agent A2
# ---------
sim_vars.agent = 2
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_2 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_2.to_pickle('gb_data/df_bic_2.pkl')
# df_bic_2.to_pickle('gb_data/df_bic_2_new.pkl')
df_bic_2.to_pickle('gb_data/df_bic_2_final.pkl')  # save data

# Agent A3
# ---------
sim_vars.agent = 3
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_3 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_3.to_pickle('gb_data/df_bic_3.pkl')
# df_bic_3.to_pickle('gb_data/df_bic_3_new.pkl')
df_bic_3.to_pickle('gb_data/df_bic_3_final.pkl')  # save data

# Agent A4
# ---------
sim_vars.agent = 4
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_4 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_4.to_pickle('gb_data/df_bic_4.pkl')
# df_bic_4.to_pickle('gb_data/df_bic_4_new.pkl')
df_bic_4.to_pickle('gb_data/df_bic_4_final.pkl')  # save data


# Agent A5
# ---------
sim_vars.agent = 5
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_5 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_5.to_pickle('gb_data/df_bic_5.pkl')
# df_bic_5.to_pickle('gb_data/df_bic_5_new.pkl')
df_bic_5.to_pickle('gb_data/df_bic_5_final.pkl')  # save data


# Agent A6
# ---------
sim_vars.agent = 6
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for model recovery
df_bic_6 = evalfun(df_subj, est_vars, eval_params, eval_simulations=True, save_plot=True)  # evaluate all models
# df_bic_6.to_pickle('gb_data/df_bic_6.pkl')
# df_bic_6.to_pickle('gb_data/df_bic_6_new.pkl')
df_bic_6.to_pickle('gb_data/df_bic_6_final.pkl')  # save data

# Save BIC as csv for Bayesian model comparison in Matlab
# -------------------------------------------------------
recov_bic_mat = np.full([len(model_params), 49], np.nan)

recov_bic_mat[:, 0] = df_bic_0[df_bic_0['agent'] == 0]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 0]['a_BIC']
recov_bic_mat[:, 1] = df_bic_0[df_bic_0['agent'] == 1]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 1]['a_BIC']
recov_bic_mat[:, 2] = df_bic_0[df_bic_0['agent'] == 2]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 2]['a_BIC']
recov_bic_mat[:, 3] = df_bic_0[df_bic_0['agent'] == 3]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 3]['a_BIC']
recov_bic_mat[:, 4] = df_bic_0[df_bic_0['agent'] == 4]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 4]['a_BIC']
recov_bic_mat[:, 5] = df_bic_0[df_bic_0['agent'] == 5]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 5]['a_BIC']
recov_bic_mat[:, 6] = df_bic_0[df_bic_0['agent'] == 6]['d_BIC'] + df_bic_0[df_bic_0['agent'] == 6]['a_BIC']

recov_bic_mat[:, 7] = df_bic_1[df_bic_1['agent'] == 0]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 0]['a_BIC']
recov_bic_mat[:, 8] = df_bic_1[df_bic_1['agent'] == 1]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 1]['a_BIC']
recov_bic_mat[:, 9] = df_bic_1[df_bic_1['agent'] == 2]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 2]['a_BIC']
recov_bic_mat[:, 10] = df_bic_1[df_bic_1['agent'] == 3]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 3]['a_BIC']
recov_bic_mat[:, 11] = df_bic_1[df_bic_1['agent'] == 4]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 4]['a_BIC']
recov_bic_mat[:, 12] = df_bic_1[df_bic_1['agent'] == 5]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 5]['a_BIC']
recov_bic_mat[:, 13] = df_bic_1[df_bic_1['agent'] == 6]['d_BIC'] + df_bic_1[df_bic_1['agent'] == 6]['a_BIC']

recov_bic_mat[:, 14] = df_bic_2[df_bic_2['agent'] == 0]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 0]['a_BIC']
recov_bic_mat[:, 15] = df_bic_2[df_bic_2['agent'] == 1]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 1]['a_BIC']
recov_bic_mat[:, 16] = df_bic_2[df_bic_2['agent'] == 2]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 2]['a_BIC']
recov_bic_mat[:, 17] = df_bic_2[df_bic_2['agent'] == 3]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 3]['a_BIC']
recov_bic_mat[:, 18] = df_bic_2[df_bic_2['agent'] == 4]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 4]['a_BIC']
recov_bic_mat[:, 19] = df_bic_2[df_bic_2['agent'] == 5]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 5]['a_BIC']
recov_bic_mat[:, 20] = df_bic_2[df_bic_2['agent'] == 6]['d_BIC'] + df_bic_2[df_bic_2['agent'] == 6]['a_BIC']

recov_bic_mat[:, 21] = df_bic_3[df_bic_3['agent'] == 0]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 0]['a_BIC']
recov_bic_mat[:, 22] = df_bic_3[df_bic_3['agent'] == 1]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 1]['a_BIC']
recov_bic_mat[:, 23] = df_bic_3[df_bic_3['agent'] == 2]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 2]['a_BIC']
recov_bic_mat[:, 24] = df_bic_3[df_bic_3['agent'] == 3]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 3]['a_BIC']
recov_bic_mat[:, 25] = df_bic_3[df_bic_3['agent'] == 4]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 4]['a_BIC']
recov_bic_mat[:, 26] = df_bic_3[df_bic_3['agent'] == 5]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 5]['a_BIC']
recov_bic_mat[:, 27] = df_bic_3[df_bic_3['agent'] == 6]['d_BIC'] + df_bic_3[df_bic_3['agent'] == 6]['a_BIC']

recov_bic_mat[:, 28] = df_bic_4[df_bic_4['agent'] == 0]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 0]['a_BIC']
recov_bic_mat[:, 29] = df_bic_4[df_bic_4['agent'] == 1]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 1]['a_BIC']
recov_bic_mat[:, 30] = df_bic_4[df_bic_4['agent'] == 2]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 2]['a_BIC']
recov_bic_mat[:, 31] = df_bic_4[df_bic_4['agent'] == 3]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 3]['a_BIC']
recov_bic_mat[:, 32] = df_bic_4[df_bic_4['agent'] == 4]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 4]['a_BIC']
recov_bic_mat[:, 33] = df_bic_4[df_bic_4['agent'] == 5]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 5]['a_BIC']
recov_bic_mat[:, 34] = df_bic_4[df_bic_4['agent'] == 6]['d_BIC'] + df_bic_4[df_bic_4['agent'] == 6]['a_BIC']

recov_bic_mat[:, 35] = df_bic_5[df_bic_5['agent'] == 0]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 0]['a_BIC']
recov_bic_mat[:, 36] = df_bic_5[df_bic_5['agent'] == 1]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 1]['a_BIC']
recov_bic_mat[:, 37] = df_bic_5[df_bic_5['agent'] == 2]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 2]['a_BIC']
recov_bic_mat[:, 38] = df_bic_5[df_bic_5['agent'] == 3]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 3]['a_BIC']
recov_bic_mat[:, 39] = df_bic_5[df_bic_5['agent'] == 4]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 4]['a_BIC']
recov_bic_mat[:, 40] = df_bic_5[df_bic_5['agent'] == 5]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 5]['a_BIC']
recov_bic_mat[:, 41] = df_bic_5[df_bic_5['agent'] == 6]['d_BIC'] + df_bic_5[df_bic_5['agent'] == 6]['a_BIC']

recov_bic_mat[:, 42] = df_bic_6[df_bic_6['agent'] == 0]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 0]['a_BIC']
recov_bic_mat[:, 43] = df_bic_6[df_bic_6['agent'] == 1]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 1]['a_BIC']
recov_bic_mat[:, 44] = df_bic_6[df_bic_6['agent'] == 2]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 2]['a_BIC']
recov_bic_mat[:, 45] = df_bic_6[df_bic_6['agent'] == 3]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 3]['a_BIC']
recov_bic_mat[:, 46] = df_bic_6[df_bic_6['agent'] == 4]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 4]['a_BIC']
recov_bic_mat[:, 47] = df_bic_6[df_bic_6['agent'] == 5]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 5]['a_BIC']
recov_bic_mat[:, 48] = df_bic_6[df_bic_6['agent'] == 6]['d_BIC'] + df_bic_6[df_bic_6['agent'] == 6]['a_BIC']

# np.savetxt('gb_data/recov_bic_mat_new.csv', recov_bic_mat, delimiter=',')
# np.savetxt('gb_data/recov_bic_mat.csv', recov_bic_mat, delimiter=',')
# todo: achtung, kontrollieren, ob "frecov_bic_mat" funktioniert
# np.savetxt('gb_data/recov_bic_mat_final.csv', recov_bic_mat, delimiter=',')  # save data
np.savetxt('gb_data/recov_bic_mat_final.csv', f=recov_bic_mat, delimiter=',')  # save data

# -------------------------------
# 2. Run parameter recovery study
# -------------------------------

# Sigma recovery
# --------------
agent = 1
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)  # generate data for parameter recovery
est_vars.experiment = 3
est_vars.agent = agent
est_vars.n_sp = 1
est_vars.n_ker = 4
est_vars.T = 25
est_vars.B = 12
est_vars.rand_sp = False
est_vars.est_sigma = True
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_sigma = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_sigma.loc[:, 'which_param_sigma'] = model_params['sigma']
recov_params_sigma.loc[:, 'sigma'] = results_df['minimum'].copy()
recov_params_sigma.loc[:, 'sigma_bias'] = results_df['minimum'] - model_params['sigma']
# recov_params.to_pickle('gb_data/param_recov_new_sigma.pkl')
recov_params_sigma.to_pickle('gb_data/param_recov_final_sigma.pkl')


# Agent A1
# --------
agent = 1
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.est_sigma = False
est_vars.agent = agent
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_a1 = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_a1.loc[:, 'which_param_beta'] = model_params['beta_A1']
recov_params_a1.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params_a1.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A1']
# recov_params.to_pickle('gb_data/param_recov_new_1.pkl')
recov_params_a1.to_pickle('gb_data/param_recov_final_1.pkl')

# Agent A2
# --------
agent = 2
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.agent = agent
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_a2 = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_a2.loc[:, 'which_param_beta'] = model_params['beta_A2']
recov_params_a2.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params_a2.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A2']
# recov_params.to_pickle('gb_data/param_recov_new_2.pkl')
recov_params_a2.to_pickle('gb_data/param_recov_final_2.pkl')


# Agent A3
# --------
agent = 3
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.agent = 3
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_a3 = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_a3.loc[:, 'which_param_beta'] = model_params['beta_A3']
recov_params_a3.loc[:, 'which_param_lambda'] = model_params['lambda_A3']
recov_params_a3.loc[:, 'lambda'] = results_df['minimum_lambda'].copy()
recov_params_a3.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params_a3.loc[:, 'lambda_bias'] = results_df['minimum_lambda'] - model_params['lambda_A3']
recov_params_a3.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A3']
# recov_params.to_pickle('gb_data/param_recov_new_3.pkl')
recov_params_a3.to_pickle('gb_data/param_recov_final_3.pkl')

# Agent A4
# ---------
agent = 4
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.agent = agent
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_a4 = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_a4.loc[:, 'which_param_beta'] = model_params['beta_A4']
recov_params_a4.loc[:, 'which_param_alpha'] = model_params['alpha_A4']
recov_params_a4.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params_a4.loc[:, 'alpha'] = results_df['minimum_alpha'].copy()
recov_params_a4.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A4']
recov_params_a4.loc[:, 'alpha_bias'] = results_df['minimum_alpha'] - model_params['alpha_A4']
# recov_params.to_pickle('gb_data/param_recov_new_4.pkl')
recov_params_a4.to_pickle('gb_data/param_recov_final_4.pkl')

# Agent A5
# --------
agent = 5
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.agent = agent
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params.loc[:, 'which_param_beta'] = model_params['beta_A5']
recov_params.loc[:, 'which_param_alpha'] = model_params['alpha_A5']
recov_params.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params.loc[:, 'alpha'] = results_df['minimum_alpha'].copy()
recov_params.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A5']
recov_params.loc[:, 'alpha_bias'] = results_df['minimum_alpha'] - model_params['alpha_A5']
recov_params.to_pickle('gb_data/param_recov_final_5.pkl')
# recov_params.to_pickle('gb_data/param_recov_new_5.pkl')

# Agent A6
# ---------
agent = 6
sim_vars.agent = agent
df_subj = recovsim(task_vars, sim_vars, exp3_data, opt_params=model_params)
est_vars.agent = agent
results_df = parallelest(df_subj, est_vars, fixed_params=model_params)
recov_params_a6 = pd.DataFrame(index=range(0, len(model_params)), dtype='float')
recov_params_a6.loc[:, 'which_param_beta'] = model_params['beta_A6']
recov_params_a6.loc[:, 'which_param_lambda'] = model_params['lambda_A6']
recov_params_a6.loc[:, 'which_param_alpha'] = model_params['alpha_A6']
recov_params_a6.loc[:, 'lambda'] = results_df['minimum_lambda'].copy()
recov_params_a6.loc[:, 'beta'] = results_df['minimum_beta'].copy()
recov_params_a6.loc[:, 'alpha'] = results_df['minimum_alpha'].copy()
recov_params_a6.loc[:, 'lambda_bias'] = results_df['minimum_lambda'] - model_params['lambda_A6']
recov_params_a6.loc[:, 'beta_bias'] = results_df['minimum_beta'] - model_params['beta_A6']
recov_params_a6.loc[:, 'alpha_bias'] = results_df['minimum_alpha'] - model_params['alpha_A6']
# recov_params.to_pickle('gb_data/param_recov_new_6.pkl')
recov_params_a6.to_pickle('gb_data/param_recov_final_6.pkl')
