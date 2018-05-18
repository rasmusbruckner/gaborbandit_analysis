# This script implements the posterior predictive check to compare the agents to participant behavior

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from gb_task_agent_int import gb_task_agent_int
import pickle
from time import sleep
from tqdm import tqdm


# Set random number generator for reproducible results
np.random.seed(123)

exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Load participants parameters
sub_params = pd.read_pickle('gb_data/modelbased.pkl')

# Task parameters
task_vars = TaskVars()
task_vars.T = 25
task_vars.B = 12

# Agent parameters
agent_vars = AgentVars()
sim_agents = [0, 1, 2, 3]

# Simulation parameters
sim_vars = SimVars()
sim_vars.take_pd = 0
n_eval = 10

# Initialize arrays
mean_ev = np.full([len(sim_agents), len(sub_params)*n_eval, 25], np.nan)
mean_corr = np.full([len(sim_agents), len(sub_params)*n_eval, 25], np.nan)

# Initialize progress bar
sleep(0.1)
print('\nRunning posterior predictive check')
sleep(0.1)
pbar = tqdm(total=n_eval*len(sub_params)*len(sim_agents))


for a in range(0, len(sim_agents)):

    # Initialize counter
    counter = -1

    agent_vars.agent = sim_agents[a]

    # Cycle over number of evaluations
    for i in range(0, n_eval):

        # Cycle over participants
        for s in range(0, len(sub_params)):

            # Update counter
            counter = counter + 1

            # Select current participant parameters
            agent_vars.sigma = sub_params['sigma_exp3'][s]
            if agent_vars.sigma < 0.015:
                agent_vars.sigma = 0.015
            agent_vars.beta = sub_params['beta'][s]
            agent_vars.lambda_param = sub_params['lambda'][s]

            # Initialize data frame for simulation
            df_subj = pd.DataFrame()

            # Cycle over task blocks
            for b in range(0, task_vars.B):

                # Block number definition
                sim_vars.block = b

                real_outc = exp3_data[(exp3_data['id'] == s) & (exp3_data['blockNumber'] == b)].copy()
                real_outc.loc[:, 'trial'] = np.linspace(0, len(real_outc) - 1, len(real_outc))
                real_outc = real_outc.set_index('trial')

                # Single block task-agent-interaction simulation
                df_block = gb_task_agent_int(task_vars, agent_vars, sim_vars, real_outc=real_outc)

                # Add data to data frame
                df_subj = df_subj.append(df_block, ignore_index=True)

            # Update progress bar
            pbar.update(1)

            # Compute mean expected value and performance
            mean_ev[a, counter, :] = df_subj.groupby(df_subj['t'])['e_mu_t'].mean()
            mean_corr[a, counter, :] = df_subj.groupby(df_subj['t'])['corr'].mean()

            # Plot all expected values
            plt.figure(a)
            x = np.linspace(0, task_vars.T - 1, task_vars.T)
            plt.plot(x, mean_ev[a, counter, :])
            plt.xlabel('Trial')
            plt.ylabel('Expected Value')

    # Plot all expected values
    plt.figure(len(sim_agents)+1)
    x = np.linspace(0, task_vars.T - 1, task_vars.T)
    plt.plot(x, np.mean(mean_ev[a, :], 0))
    plt.ylim([0, 1])
    plt.xlabel('Trial')
    plt.ylabel('Expected Value')

    # Plot p(correct)
    plt.figure(len(sim_agents)+2)
    x = np.linspace(0, task_vars.T - 1, task_vars.T)
    plt.plot(x, np.mean(mean_corr[a, :], 0))
    plt.ylim([0.4, 1])
    plt.xlabel('Trial')
    plt.ylabel('p(correct)')

# Close progress bar
pbar.close()

# Save data
f = open('gb_data/postpred.pkl', 'wb')
pickle.dump(mean_corr, f)
f.close()

plt.show()
