import numpy as np


class Task:
    # This class specifies the instance variables and methods of the Task object that models the Gabor-bandit task

    def __init__(self, task_vars):
        """ This function defines the instance variable unique to each instance

        :param task_vars: Object with task parameters
        """

        # Set instance variables based on task parameters
        self.T = task_vars.T
        self.Theta = task_vars.Theta
        self.kappa = task_vars.kappa
        self.mu = task_vars.mu
        self.experiment = task_vars.experiment

        # Compute fixed instance variables
        self.C = np.linspace(-self.kappa, self.kappa, num=20)  # range of discrete contrast differences
        self.p_cs = np.full([len(self.C), 2], np.nan)  # initialize vector for p(c_t|s_t)
        self.p_cs[:, 0] = (self.C < 0) / sum(self.C < 0)  # p(c_t|s_t = 0)
        self.p_cs[:, 1] = (self.C >= 0) / sum(self.C >= 0)  # p(c_t|s_t = 1)
        self.p_ras = np.full([2, 2], np.nan)  # initialize vector for p^(a_t)(r_t|s, \mu)
        self.p_ras[0, 0] = task_vars.mu  # p^(a_t = 0)(r_t = 1|s = 0, \mu)
        self.p_ras[0, 1] = 1 - task_vars.mu  # p^(a_t = 0)(r_t = 1|s = 1, 1-\mu)
        self.p_ras[1, 0] = 1 - task_vars.mu  # p^(a_t = 1)(r_t = 1|s = 0, 1-\mu)
        self.p_ras[1, 1] = task_vars.mu  # p^(a_t = 1)(r_t = 1|s = 1, \mu)

        # Initialize other instance variables
        self.s_t = np.full(1, np.nan)
        self.c_t = np.full(1, np.nan)
        self.r_t = np.full(1, np.nan)

    def state_sample(self):
        # This function samples the task state from a Bernoulli distribution

        # Eq. 2
        self.s_t = np.random.binomial(1, self.Theta)

    def contrast_sample(self):
        """ This function samples the contrast difference

        Depending on the specified experiment, the observation is sampled from a uniform distribution or
        element of set {-0.08, 0.08}.
        """

        # Eq. 3
        if self.experiment == 1 or self.experiment == 3:
            p_cs_giv_s = self.p_cs[:, self.s_t].flatten().tolist()  # contrast differences conditional on state
            s_cs_giv_s = np.random.multinomial(1, p_cs_giv_s, size=1).flatten()  # sample contrast difference
            i_cs_giv_s = np.nonzero(s_cs_giv_s)  # index of sampled contrast difference
            self.c_t = self.C[i_cs_giv_s]  # select contrast difference according to index
        else:
            if self.s_t == 0:
                self.c_t = self.C[0]  # most negative contrast difference
            else:
                self.c_t = self.C[-1]  # most positive contrast difference

    def reward_sample(self, a_t):
        """ This function samples the reward

        :param a_t: Current economic decision
        """

        # Eq. 4
        self.r_t = np.random.binomial(1, self.p_ras[a_t, self.s_t])
