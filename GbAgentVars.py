import numpy as np


class AgentVars:
    # This class specifies the agent parameters of agent A0, A1, A2, A3 and A4 (GbAgent.py)

    def __init__(self):
        """ This function defines the instance variable unique to each instance

               sigma:           Perceptual sensitivity parameter (default sigma = 0.04)
               beta:            Inverse temperature parameter of softmax choice rule (default beta = 100)
               lambda_param:    Weight of A1 as opposed to A2 in agent A3
               c0:              Uniform prior distribution over mu
               kappa:           Maximal contrast difference value (default kappa = 0.08)
               agent:           Current agent-based computational model (default agent = 1)
               eval_ana:        Evaluate agent analytically (1) or numerically (0)

        """

        self.sigma = 0.04
        self.beta = 100
        self.lambda_param = np.nan
        self.c0 = np.array([1])
        self.kappa = 0.08
        self.agent = 1
        self.eval_ana = 1
