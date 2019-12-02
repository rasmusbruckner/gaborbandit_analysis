class TaskVars:
    # This class specifies the task parameters of the Gabor-bandit task (GbTask.py)

    def __init__(self):
        """ This function defines the instance variable unique to each instance

            T: Number of trials
            B: Number of blocks
            Theta: Determines parameter in p(s_t; Theta)
            kappa: Maximal contrast difference value
            mu: p^{a_t = 0}{r_t = 1|s_t = 0} (reward probability)
            experiment: Experiment that is simulated

        """

        self.T = 25
        self.B = 12
        self.Theta = 0.5
        self.kappa = 0.08
        self.mu = 0.8
        self.experiment = 3
