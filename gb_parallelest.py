import pandas as pd
from multiprocessing import Pool
from GbEstimation import GbEstimation
from time import sleep
from tqdm import tqdm


def parallelest(exp_data, est_vars, **kwargs):
    """ This function implements the parallel parameter estimation

    :param exp_data: Data of experimental data that are used for parameter estimation
    :param est_vars: Estimation variables object instance
    :param kwargs: optionally, fixed parameters sigma and beta
    :return: results_df: Data frame with participants IDs, estimated parameters, maximum likelihood and BIC
    """

    # If provided, get fixed parameters: sigma and beta
    fixed_params = kwargs.get('fixed_params', None)

    # Initialize estimation object
    estimation = GbEstimation(est_vars)

    # Inform user
    sleep(0.1)
    print('\nParallel parameter estimation:')
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=est_vars.n_sim)

    # Function for progress bar update
    def callback(x):
        pbar.update(1)

    # Pool object instance for parallel processing
    pool = Pool(processes=est_vars.n_ker)

    # Estimate parameters
    if est_vars.experiment == 1:
        results = [pool.apply_async(estimation.model_exp1, args=(exp_data[(exp_data['id'] == i)].copy(),),
                                    callback=callback) for i in range(0, est_vars.n_sim)]
    else:
        results = [pool.apply_async(estimation.model_exp23, args=(exp_data[(exp_data['id'] == i)].copy(),
                                                                  est_vars, fixed_params.loc[i, :]),
                                    callback=callback) for i in range(0, est_vars.n_sim)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()

    # Put estimated parameters in data frame
    if est_vars.experiment == 1:
        results_df = pd.DataFrame(output, columns=['llh', 'd_BIC', 'minimum', 'id'])
    else:
        # results_df = pd.DataFrame(output, columns=['llh', 'BIC_a', 'BIC_d', 'id', 'model', 'minimum'])
        results_df = pd.DataFrame(output, columns=['llh', 'a_BIC', 'd_BIC', 'id', 'agent', 'minimum'])

    # Make sure we keep the same order of participants
    results_df = results_df.sort_values(by=['id'])

    pbar.close()

    return results_df
