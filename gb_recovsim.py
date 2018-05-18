import numpy as np
import pandas as pd
from GbAgentVars import AgentVars
from gb_task_agent_int import gb_task_agent_int
from time import sleep
from tqdm import tqdm


def recovsim(task_vars, sim_vars, exp_data, **kwargs):
    """ This function runs the simulations for parameter and model recovery

    :param task_vars: Variables to initialize the task object
    :param sim_vars: Variables to initialize the simulation object
    :param exp_data: Experimental data
    :param kwargs: Optionally; participant parameters
    :return: df_recov: Data frame with simulated data
    """

    # Extract the optionally provided parameters
    opt_params = kwargs.get('opt_params', None)

    # Agent parameters
    agent_vars = AgentVars()
    agent_vars.agent = sim_vars.agent

    # Initialize data frame for data that will be recovered
    df_recov = pd.DataFrame()

    # Initialize counter for simulation order
    counter = -1
    sleep(0.1)
    print('\nSimulating data for recovery:')
    sleep(0.1)
    pbar = tqdm(total=sim_vars.n_sim)

    for a in range(0, len(sim_vars.param_range)):

        # Select true parameters for simulations
        # --------------------------------------
        if task_vars.experiment == 1:

            agent_vars.sigma = sim_vars.param_range[a]

        elif task_vars.experiment == 2:

            agent_vars.beta = sim_vars.param_range[a]

        elif task_vars.experiment == 3:

            agent_vars.lambda_param = sim_vars.param_range[a]

        # Task-agent interaction simulations
        # ----------------------------------
        for m in range(0, sim_vars.N):

            # For experiment 2 and 3 we additionally participant parameters
            if task_vars.experiment == 2:

                agent_vars.sigma = opt_params['sigma'][m]

            elif task_vars.experiment == 3:

                agent_vars.sigma = opt_params['sigma'][m]
                agent_vars.beta = opt_params['beta'][m]

            counter += 1

            # Cycle over task blocks
            for b in range(0, task_vars.B):

                # Extract data of current block
                exp_data_block = exp_data[(exp_data['id'] == m) & (exp_data['b_t'] == b)].copy()

                # Add trial as index
                exp_data_block.loc[:, 'trial'] = np.linspace(0, len(exp_data_block) - 1, len(exp_data_block))
                exp_data_block = exp_data_block.set_index('trial')
                # Block number definition
                sim_vars.block = b

                # Single block task-agent-interaction simulation
                df_block = gb_task_agent_int(task_vars, agent_vars, sim_vars, real_outc=exp_data_block)
                df_block['id'] = counter
                df_block['blockNumber'] = b

                # Add data to data frame
                df_recov = df_recov.append(df_block, ignore_index=True)

            # Update progress bar
            pbar.update(1)

    if task_vars.experiment == 2 or task_vars.experiment == 3:

        df_recov = df_recov.rename(index=str, columns={"corr": "decision2.corr"})
        df_recov['a_t'] = df_recov['a_t'] + 1

    # Close progress par
    pbar.close()

    return df_recov
