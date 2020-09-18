class TaskVars:
    # This class specifies the task parameters of the Gabor-bandit task (GbTask.py)

    def __init__(self):
        # This function defines the instance variable unique to each instance

        self.T = 25  # number of trials
        self.B = 12  # number of blocks
        self.Theta = 0.5  # determines parameter in p(s_t; Theta)
        self.kappa = 0.08  # maximal contrast difference value
        self.mu = 0.8  # p^{a_t = 0}{r_t = 1|s_t = 0} (reward probability)
        self.experiment = 3  # experiment that is simulated
