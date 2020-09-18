import numpy as np


class SimVars:
    # This class specifies the simulation parameters

    def __init__(self):
        # This function defines the instance variable unique to each instance

        self.take_pd = 0  # indicates if perceptual decisions of participants will be used in simulations
        self.N = 1  # number of participants
        self.n_sim = 1  # number of simulations
        self.block = 0  # current block number
        self.agent = 1  # current agent-based computational model
        self.param_range = np.nan  # range of parameter
