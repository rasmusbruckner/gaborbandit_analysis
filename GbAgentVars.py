import numpy as np


class AgentVars:
    # This class specifies the agent parameters of agent A0, A1, A2, A3 and A4 (GbAgent.py)

    def __init__(self):
        """ This function defines the instance variable unique to each instance

               set_0: Set of observations
               sigma: Perceptual sensitivity parameter
               beta: Inverse temperature parameter of softmax choice rule
               lambda_param: Weight of A1 vs. A2 in agent A3 and A4 vs. A5 in agent A6
               c0: Uniform prior distribution over mu
               q_0_0: XXXXXXXXXXXXXXXXXXX
               kappa: Maximal contrast difference value
               agent: Current agent-based computational model
               eval_ana: Evaluate agent analytically (1) or numerically (0)
               alpha: Learning rate parameter from A4 and A5
               task_agent_analysis: XXXXXXXXXXXXXXXXXX

        """

        self.set_o = np.linspace(-0.2, 0.2, num=20)
        self.sigma = 0.04
        self.beta = 100
        self.lambda_param = np.nan
        self.c0 = np.array([1])
        self.q_0_0 = 0.5
        self.kappa = 0.08
        self.agent = 1
        self.eval_ana = 1
        self.alpha = 0.1
        self.task_agent_analysis = np.nan
