import numpy as np


class SimVars:
    # This class specifies the simulation parameters

    def __init__(self):
        """ This function defines the instance variable unique to each instance

        take_pd:    Indicates if perceptual decisions of participants will be used in simulations
        N:          Number of participants
        n_sim:         Number of simulations
        block:      Currnent block number
        agent:      Current agent-based computational model
        param:      Range of parameter
        """

        self.take_pd = 0
        self.N = 1
        self.n_sim = 1
        self.block = 0
        self.agent = 1
        self.param_range = np.nan