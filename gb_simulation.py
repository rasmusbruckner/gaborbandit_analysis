import numpy as np
import pandas as pd
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from gb_task_agent_int import gb_task_agent_int
from time import sleep
from tqdm import tqdm


def gb_simulation(T, B, sigma, agent, beta):
    """ This function runs the simulations for model demonstration

    :param T:       Number of trials
    :param B:       Number of blocks
    :param sigma:   Perceptual sensitivity
    :param agent:   Agent-based computational model
    :param beta:    Inverse temperature parameter
    :return: df_subj: Data frame containing simulated data
    """

    # Parameter definition
    # --------------------
    # Task parameters in GbTaskVars.py object
    task_vars = TaskVars()
    task_vars.T = T
    task_vars.B = B
    task_vars.experiment = 3  

    # Agent parameters in GbAgentVars.py object
    agent_vars = AgentVars()
    agent_vars.sigma = sigma
    agent_vars.beta = beta
    agent_vars.c0 = np.array([1])
    agent_vars.kappa = task_vars.kappa 
    agent_vars.agent = agent
    agent_vars.lambda_param = np.nan

    # Simulation parameters in GbSimVars.py object
    sim_vars = SimVars()

    # Task-agent interaction simulations and evaluations
    # --------------------------------------------------

    # Initialize data frame for simulation
    df_subj = pd.DataFrame()

    sleep(0.1)
    print('\nSimulating data for agent demonstration:')
    sleep(0.1)
    pbar = tqdm(total=task_vars.B)

    # Cycle over task blocks
    for b in range(0, task_vars.B):

        # Update progress bar
        pbar.update(1)

        # Block number definition
        sim_vars.block = b
        sim_vars.take_pd = 0

        # Single block task-agent-interaction simulation
        df_block = gb_task_agent_int(task_vars, agent_vars, sim_vars)

        # Add data to data frame
        df_subj = df_subj.append(df_block, ignore_index=True)

    pbar.close()

    return df_subj
