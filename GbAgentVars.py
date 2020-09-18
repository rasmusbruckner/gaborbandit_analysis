import numpy as np


class AgentVars:
    # This class specifies the agent parameters of agent A0, A1, A2, A3, A4, A6, and A6 (GbAgent.py)

    def __init__(self):
        # This function defines the instance variable unique to each instance

        self.set_o = np.linspace(-0.2, 0.2, num=20)  # set of observations
        self.sigma = 0.04  # perceptual sensitivity parameter
        self.beta = 100  # inverse temperature parameter of softmax choice rule
        self.lambda_param = np.nan  # weight of A1 vs. A2 in agent A3 and A4 vs. A5 in agent A6
        self.c0 = np.array([1])  # uniform prior distribution over mu
        self.q_0_0 = 0.5  # contingency parameter of Q-learning models
        self.kappa = 0.08  # maximal contrast difference value
        self.agent = 1  # current agent-based computational model
        self.eval_ana = 1  # evaluate agent analytically (1) or numerically (0)
        self.alpha = 0.1  # learning-rate parameter from A4 and A5
        self.task_agent_analysis = np.nan  # if true, integration over observations is turned-on
