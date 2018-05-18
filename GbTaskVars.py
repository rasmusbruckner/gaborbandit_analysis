class TaskVars:
    # This class specifies the task parameters of the Gabor-bandit task (GbTask.py)

    def __init__(self):
        """ This function defines the instance variable unique to each instance

            T:          Number of trials (default T = 25)
            B:          Number of blocks (default B = 12)
            Theta:      Determines parameter in p(s_t; Theta) (default Theta = 0.5)
            kappa:      Maximal contrast difference value (default kappa = 0.08)
            mu:         p^{a_t = 0}{r_t = 1|s_t = 0} (default mu = 0.8)
            experiment: Experiment that is simulated

        """

        self.T = 25
        self.B = 12
        self.Theta = 0.5
        self.kappa = 0.08
        self.mu = 0.8
        self.experiment = 3