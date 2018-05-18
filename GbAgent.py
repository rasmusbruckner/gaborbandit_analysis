import numpy as np
from scipy.stats import norm
import scipy.stats as stats


class Agent:
    # This class specifies the instance variables and methods of the Agent object that models A0 - A4

    def __init__(self, agent_vars):
        """ This function defines the instance variable unique to each instance

                                All attributes of agent_vars object
             O:                 Discrete set of observations
             o_t:               Current observation
             d_t:               Current perceptual decision
             a_t:               Current economic decision
             r_t:               Current reward
             pi_0:              Belief state in favor of s_t = 0: p(s_t = 0|o_t)
             pi_1:              Belief state in favor of s_t = 1: p(s_t = 1|o_t)
             pi_t:              Vector containing both state probabilities
             E_mu_t:            Expected value
             v_a_0:             Valence a_t = 0
             v_a_1:             Valence a_t = 1
             v_a_t:             Vector containing both valences
             v_a_t_A1:          Vector containing both valences of A1
             v_a_t_A2:          Vector containing both valences of A2
             p_a_t:             Vector containing choice probabilites
             p_a_t_A1:          Vector containing choice probabilites of A1
             p_a_t_A2:          Vector containing choice probabilites of A2
             t:                 Degree of polynomial
             p_o_giv_u:         Probabilities of observations given contrast differences
             p_o_giv_u_norm:    Normalized probabilities of observations given contrast differences
             C:                 Polynomial coefficients
             q_0:               Fraction for polynomial update
             q_1:               Fraction for polynomial update
             d:                 Temporary, trial-specific polynomial coefficients
             G:                 Gamma factor
             

        :param agent_vars: Object with agent parameters
        """

        # Set instance variables based on agent parameters
        self.kappa = agent_vars.kappa
        self.sigma = agent_vars.sigma
        self.c_t = agent_vars.c0
        self.beta = agent_vars.beta
        self.agent = agent_vars.agent
        self.lambda_param = agent_vars.lambda_param
        self.eval_ana = agent_vars.eval_ana

        # Initialize other instance variables
        self.O = np.linspace(-0.2, 0.2, num=20)
        self.o_t = np.nan
        self.d_t = np.nan
        self.a_t = np.nan
        self.r_t = np.nan
        self.pi_0 = np.nan
        self.pi_1 = np.nan
        self.pi_t = np.full(2, np.nan)
        self.E_mu_t = np.nan
        self.v_a_0 = np.nan
        self.v_a_1 = np.nan
        self.v_a_t = np.nan
        self.v_a_t_A1 = np.full(2, np.nan)
        self.v_a_t_A2 = np.full(2, np.nan)
        self.p_a_t = np.nan
        self.p_a_t_A1 = np.full(2, np.nan)
        self.p_a_t_A2 = np.full(2, np.nan)
        self.t = np.nan
        self.p_o_giv_u = np.full(len(self.O), np.nan)
        self.p_o_giv_u_norm = np.full(len(self.O), np.nan)
        self.C = np.nan
        self.q_0 = np.nan
        self.q_1 = np.nan
        self.d = np.nan
        self.G = np.nan

    def observation_sample(self, c_t):
        """ This function samples the observation conditional on the contrast difference

            Depending on the selected agent, the sampled observation is drawn from a Gaussian
            distribution (agent = 0) or equal to the presented contrast difference (agent = 1,2,3,4).

        :param c_t: Presented contrast difference
        """

        if self.agent == 4:
            self.o_t = np.random.normal(c_t, self.sigma)
        else:
            self.o_t = c_t

    def p_s_giv_o(self, o_t):

        """ This function evaluates the probability density function of the conditional distribution p(s_t|o_t)


        :param o_t:         Current observation
        :return: pi_0, pi_1: belief state p(s_t|o_t)
        """

        # Compute the Gaussian cumulative distribution functions
        u = norm.cdf(0, o_t, self.sigma)
        v = norm.cdf(-self.kappa, o_t, self.sigma)
        w = norm.cdf(self.kappa, o_t, self.sigma)

        # Compute belief state p(s_t|o_t)
        pi_0 = (u - v) / (w - v)
        pi_1 = (w - u) / (w - v)

        return pi_0, pi_1

    def decide_p(self):
        """ This function implements the agent's perceptual decision strategy

            A4 generates random choices, the other agents generate choices according to pi_1.
        """

        if self.agent == 0:
            self.pi_0 = 0.5
            self.pi_1 = 0.5
        else:
            # Compute belief state
            self.pi_0, self.pi_1 = self.p_s_giv_o(self.o_t)

        self.pi_t = [self.pi_0, self.pi_1]

        self.d_t = np.random.binomial(1, self.pi_1)

    def eval_poly(self):
        """ This function evaluates the polynomial

        :return: poly_eval: Evaluated polynomial
        """

        poly_int = np.polyint(np.append(self.c_t, [0]))  # indefinite integral of polynomial
        poly_eval = np.polyval(poly_int, [0, 1])  # evaluate polynomial in [0, 1]
        poly_eval = np.diff(poly_eval)  # difference of evaluated polynomial

        return poly_eval

    def softmax(self):
        # This function implements the softmax action selection

        self.p_a_t = np.exp(np.dot(self.v_a_t, self.beta)) / np.sum(np.exp(np.dot(self.v_a_t, self.beta)))

    def compute_valence(self, pi_0, pi_1):
        """ This function computes the action-dependent reward probability

        :param pi_0: belief in favor of s_t = 0
        :param pi_1: belief in favor of s_t = 1
        :return: v_a_t: Vector containing action valences
        """

        if self.eval_ana:
            # Conditional expected value of \mu given o_(1:t-1), r_(1:t-1)
            self.E_mu_t = self.eval_poly()
        else:
            self.E_mu_t = np.dot(self.mu, self.p_mu)

        # Action valence evaluation
        v_a_0 = (pi_0 - pi_1) * self.E_mu_t + pi_1
        v_a_1 = (pi_1 - pi_0) * self.E_mu_t + pi_0

        # Concatenate action valences
        v_a_t = [v_a_0, v_a_1]

        return v_a_t

    def decide_e(self):
        # This function implements the agent's economic choice strategy

        # elif self.agent == 4:
        if self.agent == 0:

            # Compute action valences
            self.v_a_t = [0.5, 0.5]
            self.v_a_0 = self.v_a_t[0]
            self.v_a_1 = self.v_a_t[1]

        if self.agent == 1 or self.agent == 4:

            # Compute belief state
            pi_0, pi_1 = self.p_s_giv_o(self.o_t)

            # Compute action valences
            self.v_a_t = self.compute_valence(pi_0, pi_1)
            self.v_a_0 = self.v_a_t[0]
            self.v_a_1 = self.v_a_t[1]

        elif self.agent == 2:

            # Compute belief state categorically
            if self.d_t == 0:
                pi_0 = 1
                pi_1 = 0
            else:
                pi_0 = 0
                pi_1 = 1

            # Compute action valences
            self.v_a_t = self.compute_valence(pi_0, pi_1)
            self.v_a_0 = self.v_a_t[0]
            self.v_a_1 = self.v_a_t[1]

        elif self.agent == 3:

            # Compute belief state
            pi_0, pi_1 = self.p_s_giv_o(self.o_t)

            # Compute action valences using probabilistic belief state
            v_a_t_A1 = self.compute_valence(pi_0, pi_1)

            # Compute belief state categorically
            if self.d_t == 0:
                pi_0 = 1
                pi_1 = 0
            else:
                pi_0 = 0
                pi_1 = 1

            # Compute action valences using categorical belief state
            v_a_t_A2 = self.compute_valence(pi_0, pi_1)

            # Compute action valences of mixture between both models
            self.v_a_0 = v_a_t_A1[0] * self.lambda_param + v_a_t_A2[0] * (1 - self.lambda_param)
            self.v_a_1 = v_a_t_A1[1] * self.lambda_param + v_a_t_A2[1] * (1 - self.lambda_param)
            self.v_a_t = [self.v_a_0, self.v_a_1]

        self.softmax()

        # Sample agent action: p(a_t = 1) = p_a_t(1)
        self.a_t = np.random.binomial(1, self.p_a_t[1])

    def compute_q(self, r_t, pi_0, pi_1):
        """ This function computes the q_0 and q_1 of the Gabor-bandit probabilistic model

        :param r_t:         Current reward
        :param pi_0:        Belief in favor of s_t = 0
        :param pi_1:        Belief in favor of s_t = 1
        :return: q_0, q_1:  Computed variables
        """

        # Extract the current reward
        self.r_t = r_t

        # Evaluate the degree of the resulting polynomial (ohne self?)
        self.t = self.c_t.size + 1

        # Evaluate action-dependent reward value
        self.r_t = self.r_t + (self.a_t * ((-1) ** (2 + self.r_t)))

        if self.eval_ana:
            # Evaluate the gamma factor
            self.G = self.eval_poly()

            # Evaluate common denominator of q_0 and q_1
            self.C = (pi_0 - pi_1) * ((1 - self.G) ** (1 - self.r_t)) * (
                self.G ** self.r_t) + pi_1

            # Evaluate q_0 and q_1
            q_0 = (pi_1 ** self.r_t) * (pi_0 ** (1 - self.r_t)) / self.C
            q_1 = ((-1) ** (self.r_t + 1)) * (pi_0 - pi_1) / self.C

        else:
            # Evaluate q_0 and q_1
            q_0 = (pi_1 ** self.r_t) * (pi_0 ** (1 - self.r_t))
            q_1 = ((-1) ** (self.r_t + 1)) * (pi_0 - pi_1)

        return q_0, q_1

    def integrate_q(self, r_t):
        """ This function computes the integral over q_0 and q_1 conditional on the contrast difference

        :param r_t:             Current reward
        :return: q_0, q_1:      Values of variables after integration
        """

        q_matrix = np.full([len(self.O), 2], np.nan)  # matrix containing all q's
        self.p_o_giv_u = stats.norm.pdf(self.O, self.o_t, self.sigma)  # evaluate p(o_t|u_t)
        self.p_o_giv_u_norm = self.p_o_giv_u / sum(self.p_o_giv_u)  # normalize evaluated probabilities

        for i in range(0, len(self.O)):

            pi_0, pi_1 = self.p_s_giv_o(self.O[i])
            q_0, q_1 = self.compute_q(r_t, pi_0, pi_1)
            q_matrix[i, 0] = q_0
            q_matrix[i, 1] = q_1

        q_0 = sum(q_matrix[:, 0] * self.p_o_giv_u_norm)
        q_1 = sum(q_matrix[:, 1] * self.p_o_giv_u_norm)

        return q_0, q_1

    def learn(self, r_t):
        """ This function implements the recursive Bayesian estimation of \mu

        :param r_t: Current reward
        """

        # Compute q's depending on the agent
        # ----------------------------------

        if self.agent == 1 or self.agent == 2 or self.agent == 3 or self.agent == 4:

            if self.agent == 1:

                # Compute q based on integral over possible observations
                self.q_0, self.q_1 = self.integrate_q(r_t)

            elif self.agent == 2:

                # Compute belief state based on perceptual decision
                if self.d_t == 0:
                    pi_0 = 1
                    pi_1 = 0
                else:
                    pi_0 = 0
                    pi_1 = 1

                # Compute q based on perceptual decision
                self.q_0, self.q_1 = self.compute_q(r_t, pi_0, pi_1)

            elif self.agent == 3:

                # Compute q based on integral over possible observations
                q_0_bs, q_1_bs = self.integrate_q(r_t)

                # Compute belief state based on perceptual decision
                if self.d_t == 0:
                    pi_0 = 1
                    pi_1 = 0
                else:
                    pi_0 = 0
                    pi_1 = 1

                # Compute q based on perceptual decision
                q_0_cat, q_1_cat = self.compute_q(r_t, pi_0, pi_1)

                # Compute final q as a function of \lambda
                self.q_0 = q_0_bs * self.lambda_param + q_0_cat * (1 - self.lambda_param)
                self.q_1 = q_1_bs * self.lambda_param + q_1_cat * (1 - self.lambda_param)

            if self.agent == 4:

                # Compute belief state using sampled observation
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute q using sampled observation
                self.q_0, self.q_1 = self.compute_q(r_t, pi_0, pi_1)

            # Update polynomial coefficients
            # ------------------------------

            # todo: hier vielleicht funktion draus machen

            if self.eval_ana:

                # Initialize update coefficients
                self.d = np.zeros(self.t)

                # Evaluate last element of d_t
                self.d[-1] = self.q_0 * self.c_t[-1]

                # Evaluate d_1,... (if existent)
                for n in range(0, (self.t-2)):
                    self.d[-n-2] = self.q_1 * self.c_t[-n-1] + self.q_0 * self.c_t[-n-2]

                # Evaluate first element of d_t
                self.d[0] = self.q_1 * self.c_t[0]

                # Update the coefficients
                self.c_t = self.d

        else:
            # Random choice agent (A0) constantly represents p(\mu) = 0.5
            self.G = 0.5



