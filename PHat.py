import numpy as np


class PHat:
    # This class specifies the instance variables and methods for
    # the sampling-based approximation of the pdf of p(c_t|s_t) and p_(t-1)(\mu|r_t, o_t)

    def __init__(self, nb):
        """ This function defines the instance variable unique to each instance

        o_min:              Observation bin edge minimum
        o_max:              Observation bin edge maximum
        o_nb:               Observation number of bins
        o_bw                Observation bin width
        o_e                 Observation bin edges
        o:                  Observation bin centers
        r:                  Reward bin edges = possible reward values
        r_nb:               Number of reward bin edges = number of possible reward values
        mu_min:             Task parameter bin edge minimum
        mu_max:             Task parameter bin edge maximum
        mu_nb               Task parameter number of bins
        mu_bw:              Task parameter bin width
        mu_e:               Task parameter bin edges
        mu:                 Task parameter bin centers
        p_s_giv_o_hat:      Probability density values for belief state
        S:                  Sample matrix
        p_mu_giv_r_o_hat    Probability density values for mu given reward and observation

        :param nb: Number of bins
        """

        # Approximation parameters
        self.o_min = -.1
        self.o_max = .1
        self.o_nb = nb
        self.o_bw = (self.o_max - self.o_min) / self.o_nb
        self.o_e = np.linspace(self.o_min, self.o_max, self.o_nb+1)
        self.o = self.o_e[:-1] + (self.o_bw/2)
        self.r_e = [0, 1]
        self.r_nb = len(self.r_e)
        self.mu_min = 0
        self.mu_max = 1
        self.mu_nb = nb
        self.mu_bw = (self.mu_max - self.mu_min) / self.mu_nb
        self.mu_e = np.linspace(self.mu_min, self.mu_max, self.mu_nb+1)
        self.mu = self.mu_e[:-1] + (self.mu_bw/2)
        self.p_s_giv_o_hat = np.nan
        self.S = np.nan
        self.p_mu_giv_r_o_hat = np.nan

    def gb_p_hat(self):
        # This function implements the sampling-based approximation of the pdf of p(c_t|s_t) and p_(t-1)(\mu|r_t, o_t)

        # Approximation array initialization
        self.p_s_giv_o_hat = np.full([2, self.o_nb], np.nan)

        # Cycle over observation bins
        for bo in range(0, self.o_nb):

            # Current bin of interest
            boi = [self.o_e[bo], self.o_e[bo + 1]]

            # Samples of interest: state variable realization for which the observation realization falls
            # into the current bin of interest
            soi = self.S[1, :][(self.S[3, :] > boi[0]) & (self.S[3, :] < boi[1])]

            # There may be cases of no observation in the current bin of interest
            if soi.size != 0:

                # If there are observation in the current bin of interest, count the number of 0's and 1's
                # and devide by the number of observations in the current bin
                self.p_s_giv_o_hat[:, bo] = np.append(len(soi) - sum(soi), sum(soi)) / len(soi)

        # Approximation array initialization
        # Discretized task paramter space x discretized observation space x number of rewards
        self.p_mu_giv_r_o_hat = np.full([self.mu_nb, self.o_nb, self.r_nb], np.nan)

        # Cycle over reward values
        for br in range(0, self.r_nb):

            # Cycle over observation bins
            for bo in range(0, self.o_nb):

                # Current bins of interest
                o_boi = [self.o_e[bo], self.o_e[bo + 1]]
                r_boi = self.r_e[br]

                # All samples, for which the current reward of interest was realized
                S_r = self.S[:, self.S[4, :] == r_boi]

                # ...of these samples for
                S_r_o = S_r[:, (S_r[3, :] > o_boi[0]) & (S_r[3, :] <= o_boi[1])]

                # Initialize sample count array for current reward and observation bins
                n_r_o_mu = np.full([self.mu_nb], np.nan)

                # Cycle over task parameter bins
                for bmu in range(0, self.mu_nb):

                    # Task parameter bin of interest
                    mu_boi = [self.mu_e[bmu], self.mu_e[bmu+1]]

                    # Of all S_r_o samples, consider those for which a task parameter value within the current task
                    # parameter bin of interest was realized
                    s_r_o_mu = S_r_o[:, (S_r_o[0, :] > mu_boi[0]) & (S_r_o[0, :] <= mu_boi[1])]

                    # Count how many of these samples were there
                    n_r_o_mu[bmu] = s_r_o_mu.shape[1]

                # Estimate density over task parameter by dividing the counts over task parameter values by number
                # of samples for which an observation value within the current observation bin of interest was realized
                self.p_mu_giv_r_o_hat[:, bo, br] = (1/self.mu_bw) * n_r_o_mu/S_r_o.shape[1]
