import numpy as np


class EstVars:
    # This class specifies the estimation parameters

    def __init__(self, task_vars):
        # This function defines the instance variable unique to each instance

        self.T = task_vars.T  # number of trials
        self.B = task_vars.B  # number of blocks
        self.n_sim = np.nan  # number of simulations
        self.n_sp = 1  # number of starting points
        self.n_ker = 4  # number of kernels during parallelization
        self.experiment = 1  # current experiment
        self.agent = 1  # current agent-based computational model
        self.s_bnds = (0.01, 0.1)  # estimation boundaries for sigma parameter
        self.b_bnds = (0.01, 20.0)  # estimation boundaries for beta parameter
        self.a_bnds = (0, 1)  # estimation boundaries of alpha parameter
        self.l_bnds = (0, 1)  # estimation boundaries of lambda parameter
        self.s_fixedsp = 0.02  # fixed starting point of sigma parameter
        self.b_fixedsp = 5.0  # fixed starting point of beta parameter
        self.l_fixedsp = 0.5  # fixed starting point of lambda parameter
        self.a_fixedsp = 0.2  # fixed starting point of alpha parameter
        self.rand_sp = False  # indicate if you want to use random starting points
        self.id = np.nan  # ID of current participant
        self.real_data = True  # indicate if estimation is based on real data
        self.est_sigma = False  # indicate if sigma parameter is estimated
