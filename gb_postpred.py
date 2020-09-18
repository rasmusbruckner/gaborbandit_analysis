# This script implements the posterior predictive check to compare the agents to participant behavior

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GbTaskVars import TaskVars
from GbAgentVars import AgentVars
from GbSimVars import SimVars
from run_postpred import run_postpred

# Set random number generator for reproducible results
np.random.seed(123)

# Load data of third experiment
exp3_data = pd.read_pickle('gb_data/gb_exp3_data.pkl')

# Load participants parameters
sub_params = pd.read_pickle('gb_data/modelbased.pkl')

# Task parameters
task_vars = TaskVars()
task_vars.T = 25
task_vars.B = 12

# Agent parameters
agent_vars = AgentVars()
agent_vars.task_agent_analysis = True
sim_agents = [0, 1, 2, 3, 4, 5, 6]

# Simulation parameters
sim_vars = SimVars()
sim_vars.take_pd = 0
sim_vars.n_eval = 10

# -------------------------
# 2. Posterior predictions
# -------------------------

# Turn integration over observations on
sim_vars.post_pred = True

# Run simulation
run_postpred(sub_params, exp3_data, agent_vars, task_vars, sim_vars, sim_agents)

# --------------------
# 2. Prior predictions
# --------------------

# Adjust parameters
sub_params['sigma'] = sub_params['sigma_exp3'] = 0.04
sub_params['beta_A1'] = sub_params['beta_A2'] = sub_params['beta_A3'] =\
    sub_params['beta_A4'] = sub_params['beta_A5'] = sub_params['beta_A6'] = 100
sub_params['lambda_A3'] = sub_params['lambda_A6'] = 0.5
sub_params['alpha_A4'] = sub_params['alpha_A5'] = sub_params['alpha_A6'] = 0.1

# Turn posterior predictions off
sim_vars.post_pred = False

# Simulate 1 time
sim_vars.n_eval = 1

# Turn integration over observations off
agent_vars.task_agent_analysis = False

# Run simulation
run_postpred(sub_params, exp3_data, agent_vars, task_vars, sim_vars, sim_agents)

# Show figures
plt.show()
