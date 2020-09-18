import numpy as np
import pandas as pd
from GbEstimation import GbEstimation
from tqdm import tqdm
from time import sleep


def evalfun(exp_data, est_vars, sub_params, eval_simulations=False, save_plot=0):
    """ This function evaluates the agent-based computational models

    :param exp_data: Data frame that contains current data set
    :param est_vars: Estimation variables
    :param sub_params: Participant parameters
    :param eval_simulations: Indicate if agents should be evaluated or not
    :param save_plot: Indicate if plot should be saved
    :return: df_bic: Data frame containing BICs
    """

    # Initialize estimation object
    gb_estimation = GbEstimation(est_vars)

    # Indicate that we only evaluate models
    est_params = 0

    if eval_simulations:

        # Initialize progress bar
        sleep(0.1)
        generating_agent = np.unique(exp_data['agent'])[0]
        print('\nEvaluating computational models | Agent A%s:' % generating_agent)
        sleep(0.1)
        n = len(list(set(exp_data['id'])))
        pbar = tqdm(total=n)

    # Initialize lists for model BIC's
    bic_a0 = list()
    bic_a1 = list()
    bic_a2 = list()
    bic_a3 = list()
    bic_a4 = list()
    bic_a5 = list()
    bic_a6 = list()

    # Cycle over participants
    for i in range(0, len(sub_params)):

        # Agent A0
        est_vars.agent = 0
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a0.append(output)

        # Agent A1
        est_vars.agent = 1
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a1.append(output)

        # Agent A2
        est_vars.agent = 2
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a2.append(output)

        # Agent A3
        est_vars.agent = 3
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a3.append(output)

        # Agent A4
        est_vars.agent = 4
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a4.append(output)

        # Agent A5
        est_vars.agent = 5
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a5.append(output)

        # Agent A6
        est_vars.agent = 6
        output = gb_estimation.edm_l_model(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(),
                                           est_params=est_params, save_plot=save_plot)
        bic_a6.append(output)

        if eval_simulations:
            # Update progress bar
            pbar.update(1)

    if eval_simulations:

        # Close progress bar
        pbar.close()

    # Put results into data frame
    columns = ['llh', 'd_BIC', 'a_BIC', 'id', 'agent']
    bic_a0 = pd.DataFrame(bic_a0, columns=columns)
    bic_a1 = pd.DataFrame(bic_a1, columns=columns)
    bic_a2 = pd.DataFrame(bic_a2, columns=columns)
    bic_a3 = pd.DataFrame(bic_a3, columns=columns)
    bic_a4 = pd.DataFrame(bic_a4, columns=columns)
    bic_a5 = pd.DataFrame(bic_a5, columns=columns)
    bic_a6 = pd.DataFrame(bic_a6, columns=columns)

    # Data frame containing all BIC's
    df_bic = pd.concat([bic_a0, bic_a1, bic_a2, bic_a3, bic_a4, bic_a5, bic_a6])

    return df_bic
