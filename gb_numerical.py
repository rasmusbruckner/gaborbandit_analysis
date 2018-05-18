import numpy as np
import pandas as pd
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from gb_task_agent_int import gb_task_agent_int
from time import sleep
from tqdm import tqdm


# todo: das hier mit analytisch vergleichen..!

T = 25
B = 1
sigma = 0.01
beta = 100
agent = 4

# This function runs the simulations for the model demonstration

# Parameter definition
# --------------------
# Task parameters in GbTaskVars.py object
task_vars = TaskVars()
task_vars.T = T # 25  # number of trials per block
task_vars.B = B  # number of blocks
task_vars.mu = 0.8
#task_vars.version = 0  # task version: 0 = main task; 1 = pure economic decisions making
task_vars.experiment = 3  # experiment #3

# Agent parameters in GbAgentVars.py object
agent_vars = AgentVars()
agent_vars.sigma = sigma #0.06 ** 2 #0.0233 ** 2 #0.06 ** 2  # discrimination sensitivity
agent_vars.beta = beta  # inverse temperature parameter
agent_vars.c0 = np.array([1])  # uniform prior distribution over mu
agent_vars.kappa = task_vars.kappa  # maximal contrast difference value

#agent_vars.agent = model_space[0]  # 0: d_t; 1: o_t; 2: u_t; 3: binary belief state
agent_vars.agent = agent  # 0: d_t; 1: o_t; 2: u_t; 3: binary belief state
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

######## nur zum spaß
#import matplotlib.pyplot as plt

#agent.p_mu = agent.p_mu/np.sum(agent.p_mu)
plt.plot(agent.mu, agent.p_mu)
print(np.sum(agent.p_mu))

#% probability
#mass
#function
#pmf = agent.p_mu / (np.sum(agent.p_mu))

#x_1 = np.linspace(0, 1, 1000)

ev = np.dot(agent.mu, agent.p_mu)


######
