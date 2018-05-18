import numpy as np


class EstVars:
    # This class specifies the estimation parameters

    def __init__(self, task_vars):
        """ This function defines the instance variable unique to each instance

                      T:           Number of trials
                      B:           Number of blocks
                      n_sim:       Number of simulations
                      n_sp:        Number of starting points
                      n_ker:       Number of kernels during parallelization
                      experiment:  Current experiment
                      agent:       Current agent-based computational model
                      s_bnds:      Estimation boundaries for sigma parameter
                      b_bnds:      Estimation boundaries for beta parameter
                      l_bnds:      Estimation boundaries for lambda parameter
                      s_fixedsp:   Fixed starting point for sigma parameter
                      b_fixedsp:   Fixed starting point for beta parameter
                      l_fixed:     Fixed starting point for lambda parameter
                      rand_sp      Indicate if you want to use random starting points
                      id:          Id of current participant
        """

        self.T = task_vars.T
        self.B = task_vars.B
        self.n_sim = np.nan
        self.n_sp = 1
        self.n_ker = 4
        self.experiment = 1
        self.agent = 1
        self.s_bnds = [(0.01, 0.1), ]
        self.b_bnds = [(0, 20), ]
        self.l_bnds = [(0, 1), ]
        self.s_fixedsp = np.array([0.02])
        self.b_fixedsp = np.array([5])
        self.l_fixedsp = np.array([0.5])
        self.rand_sp = False
        self.id = np.nan
