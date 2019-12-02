import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gb_task_agent_int import gb_task_agent_int
import pickle
from time import sleep
from tqdm import tqdm


def run_postpred(sub_params, exp3_data, agent_vars, task_vars, sim_vars, sim_agents):
    """ This function runs the prior- and posterior predictions

    :param sub_params:  Subject parameters
    :param exp3_data: Experimental data
    :param agent_vars: Agent variables
    :param task_vars: Task variables
    :param sim_vars: Simulation varibles
    :param sim_agents: Indicates which agents will be simulated
    """

    # Initialize arrays
    mean_ev = np.full([len(sim_agents), len(sub_params) * sim_vars.n_eval, 25], np.nan)
    mean_corr = np.full([len(sim_agents), len(sub_params) * sim_vars.n_eval, 25], np.nan)

    # Initialize progress bar
    sleep(0.1)
    if sim_vars.post_pred:
        print('\nRunning posterior predictive check')
    else:
        print('\nRunning performance simulations')
    sleep(0.1)
    pbar = tqdm(total=sim_vars.n_eval * len(sub_params) * len(sim_agents))

    # Cycle over agents
    for a in range(0, len(sim_agents)):

        # Initialize counter
        counter = -1

        agent_vars.agent = sim_agents[a]

        # Cycle over number of evaluations
        for i in range(0, sim_vars.n_eval):

            # Cycle over participants
            for s in range(0, len(sub_params)):

                # Update counter
                counter = counter + 1

                # Select current participant parameters
                agent_vars.sigma = sub_params['sigma_exp3'][s]

                # Select agent dependent parameters
                if sim_agents[a] == 0:
                    agent_vars.beta = np.nan
                    agent_vars.alpha = np.nan
                    agent_vars.lambda_param = np.nan
                elif sim_agents[a] == 1:
                    agent_vars.beta = sub_params['beta_A1'][s]
                elif sim_agents[a] == 2:
                    agent_vars.beta = sub_params['beta_A2'][s]
                elif sim_agents[a] == 3:
                    agent_vars.lambda_param = sub_params['lambda_A3'][s]
                    agent_vars.beta = sub_params['beta_A3'][s]
                elif sim_agents[a] == 4:
                    agent_vars.alpha = sub_params['alpha_A4'][s]
                    agent_vars.beta = sub_params['beta_A4'][s]
                elif sim_agents[a] == 5:
                    agent_vars.alpha = sub_params['alpha_A5'][s]
                    agent_vars.beta = sub_params['beta_A5'][s]
                elif sim_agents[a] == 6:
                    agent_vars.alpha = sub_params['alpha_A6'][s]
                    agent_vars.lambda_param = sub_params['lambda_A6'][s]
                    agent_vars.beta = sub_params['beta_A6'][s]

                # Initialize data frame for simulation
                df_subj = pd.DataFrame()

                # Cycle over task blocks
                for b in range(0, task_vars.B):
                    # Block number definition
                    sim_vars.block = b

                    # Extract task outcomes from empirical data
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
        plt.figure(len(sim_agents) + 1)
        x = np.linspace(0, task_vars.T - 1, task_vars.T)
        plt.plot(x, np.mean(mean_ev[a, :], 0))
        plt.ylim([0, 1])
        plt.xlabel('Trial')
        plt.ylabel('Expected Value')

        # Plot probability correct
        plt.figure(len(sim_agents) + 2)
        x = np.linspace(0, task_vars.T - 1, task_vars.T)
        plt.plot(x, np.mean(mean_corr[a, :], 0))
        plt.ylim([0.4, 1])
        plt.xlabel('Trial')
        plt.ylabel('p(correct)')

    # Close progress bar
    pbar.close()

    if sim_vars.post_pred:
        # Save data
        # f = open('gb_data/postpred_april_beta.pkl', 'wb')
        f = open('gb_data/postpred_final.pkl', 'wb')
        # f = open('gb_data/predictions.pkl', 'wb')
    else:
        f = open('gb_data/predictions_final.pkl', 'wb')
    pickle.dump(mean_corr, f)
    f.close()
