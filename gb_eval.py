import pandas as pd
from GbEstimation import GbEstimation
from tqdm import tqdm
from time import sleep


def evalfun(exp_data, est_vars, sub_params, eval_a3=0):
    """ This function evaluates the agent-based computational models

    :param exp_data:        Data frame that contains current data set
    :param est_vars:        Estimation variables
    :param sub_params:      Participant parameters
    :param eval_a3:         Indicate if agent A3 should be evaluated or not
    :return: df_bic         Data frame containing BICs
    """

    # Initialize estimation object
    gb_estimation = GbEstimation(est_vars)

    # Compute number of subjects
    N = len(sub_params)

    # Indicate that we only evaluate models
    est_params = 0

    if eval_a3 == 1:
        # If we only evaluate a3 for model recovery purposes, we fix the lambda parameter to 0.5
        sub_params['lambda'] = 0.5

    # Initialize progress bar
    sleep(0.1)
    print('\nEvaluating computational models:')
    sleep(0.1)
    pbar = tqdm(total=est_vars.n_sim)

    # Evaluate categorical model
    # --------------------------
    bic_a0 = list()
    bic_a1 = list()
    bic_a2 = list()
    bic_a3 = list()

    for i in range(0, N):

        # Evaluate Agent models
        # ----------------------------
        est_vars.agent = 0
        # fixed_params = sub_params.copy()
        # fixed_params['beta'] = 0
        output = gb_estimation.model_exp23(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(), est_params)
        bic_a0.append(output)

        est_vars.agent = 1
        output = gb_estimation.model_exp23(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(), est_params)
        bic_a1.append(output)

        est_vars.agent = 2
        output = gb_estimation.model_exp23(exp_data.loc[exp_data['id'] == i, :].copy(),
                                           est_vars, sub_params.loc[i, :].copy(), est_params)
        bic_a2.append(output)

        if eval_a3 == 1:
            est_vars.agent = 3
            output = gb_estimation.model_exp23(exp_data.loc[exp_data['id'] == i, :].copy(),
                                               est_vars, sub_params.loc[i, :].copy(), est_params)
            bic_a3.append(output)

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Put results into data frame
    columns = ['llh', 'd_BIC', 'a_BIC', 'id', 'agent']
    bic_a0 = pd.DataFrame(bic_a0, columns=columns)
    bic_a1 = pd.DataFrame(bic_a1, columns=columns)
    bic_a2 = pd.DataFrame(bic_a2, columns=columns)
    bic_a3 = pd.DataFrame(bic_a3, columns=columns)

    # Data frame containing all BIC's
    df_bic = pd.concat([bic_a0, bic_a1, bic_a2, bic_a3])

    return df_bic
