import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import warnings


# todo: reset_index() anstatt andere lösung!  --> https://cmdlinetips.com/2018/04/how-to-reset-index-in-pandas-dataframe/

class Agent:
    # This class specifies the instance variables and methods of the Agent object that models A0 - A4

    def __init__(self, agent_vars):
        """ This function defines the instance variable unique to each instance

             All attributes of agent_vars object
             set_o: Discrete set of observations
             o_t: Current observation
             d_t: Current perceptual decision
             a_t: Current economic decision
             r_t: Current reward
             pi_0: Belief state in favor of s_t = 0: p(s_t = 0|o_t)
             pi_1: Belief state in favor of s_t = 1: p(s_t = 1|o_t)
             pi_t: Vector containing both state probabilities
             E_mu_t: Expected value
             v_a_0: Valence a_t = 0
             v_a_1: Valence a_t = 1
             v_a_t: Vector containing both valences
             v_a_t_A1: Vector containing both valences of A1
             v_a_t_A2: Vector containing both valences of A2
             p_a_t: Vector containing choice probabilites
             p_a_t_A1: Vector containing choice probabilites of A1
             p_a_t_A2: Vector containing choice probabilites of A2
             t: Degree of polynomial
             p_o_giv_u: Probabilities of observations given contrast differences
             p_o_giv_u_norm: Normalized probabilities of observations given contrast differences
             C: Polynomial coefficients
             q_0: Fraction for polynomial update
             q_1: Fraction for polynomial update
             d: Temporary, trial-specific polynomial coefficients
             G: Gamma factor
             

        :param agent_vars: Object with agent parameters
        """

        # Set instance variables based on agent parameters
        self.set_o = agent_vars.set_o
        self.kappa = agent_vars.kappa
        self.sigma = agent_vars.sigma
        self.c_t = agent_vars.c0
        self.beta = agent_vars.beta
        self.agent = agent_vars.agent
        self.lambda_param = agent_vars.lambda_param
        self.eval_ana = agent_vars.eval_ana
        self.alpha = agent_vars.alpha
        self.task_agent_analysis = agent_vars.task_agent_analysis

        # Initialize other instance variables
        self.o_t = np.nan
        self.d_t = np.nan
        self.a_t = np.nan
        self.r_t = np.nan
        self.pi_0 = np.nan
        self.pi_1 = np.nan
        self.E_mu_t = 0.5
        self.q_0_0 = agent_vars.q_0_0
        self.v_a_t = np.nan
        self.v_a_t_A1 = np.full(2, np.nan)
        self.v_a_t_A2 = np.full(2, np.nan)
        self.v_a_0 = np.nan
        self.v_a_1 = np.nan
        self.p_a_t = np.nan
        self.p_a_t_A1 = np.full(2, np.nan)
        self.p_a_t_A2 = np.full(2, np.nan)
        self.t = np.nan
        self.p_o_giv_u = np.full(len(self.set_o), 0.0)
        self.p_o_giv_u_norm = np.full(len(self.set_o), 0.0)
        self.C = np.nan
        self.q_0 = np.nan
        self.q_1 = np.nan
        self.d = np.nan
        self.G = np.nan
        self.p_d_t = [np.nan, np.nan]

        # initialize Q for new model
        self.Q_t = np.full([2, 2], 0.5)

        # todo: achtung für numerical
        self.product = np.ones(100)
        self.mu = np.repeat(1, 100)
        self.p_mu = np.linspace(0, 1, 100)
        self.mu_for_ev = self.mu
        self.mu_for_ev = self.mu_for_ev/np.sum(self.mu_for_ev)

    def observation_sample(self, c_t):
        """ This function samples the observation conditional on the contrast difference

            Depending on the the analysis (tak-agent-data analysis vs. simulations),
            the sampled observation is drawn from a Gaussian distribution or equal to
            the presented contrast difference.

        :param c_t: Presented contrast difference
        """

        if self.task_agent_analysis:
            self.o_t = c_t
        else:
            self.o_t = np.random.normal(c_t, self.sigma)

    def p_s_giv_o(self, o_t, avoid_imprecision=True):
        """ This function evaluates the probability density function of the conditional distribution p(s_t|o_t)

        :param o_t: Current observation
        :param avoid_imprecision: Avoid numerical underflow during belief state computation
        :return: pi_0, pi_1: belief state p(s_t|o_t)
        """

        # Compute the Gaussian cumulative distribution functions
        u = norm.cdf(0, o_t, self.sigma)
        v = norm.cdf(-self.kappa, o_t, self.sigma)
        w = norm.cdf(self.kappa, o_t, self.sigma)

        # Compute belief state p(s_t|o_t)
        if avoid_imprecision and self.sigma <= 0.015 and o_t == -0.2:

            # In participant 3 and 8, u==v==w==1 if o_t = -0.2. This is because the two participants
            # have a very low sigma parameter with which Python cannot deal anymore. In this case, we adjust pi_s
            pi_0 = 1.0
            pi_1 = 0.0

        else:

            # For all other cases, pi_s is straightforwardly computed
            pi_0 = (u - v) / (w - v)
            pi_1 = (w - u) / (w - v)

        return pi_0, pi_1

    def cat_bs(self):
        """ This function computes the categorical belief states based of the current perceptual decision

        :return: pi_0, pi_1: Categorical belief states
        """

        if self.d_t == 0:
            pi_0 = 1
            pi_1 = 0
        else:
            pi_0 = 0
            pi_1 = 1

        return pi_0, pi_1

    def decide_p(self):
        """ This function implements the agent's perceptual decision strategy

            For the tak-agent-data analysis model, we sample perceptual choices from a Gaussian CDF.
            During simulations, we minimize the squared loss and take the more likely belief state.
            The random choice agent A0 generates random choices.
        """

        if self.agent == 0:
            self.pi_0 = 0.5
            self.pi_1 = 0.5
            p_d_0 = 0.5
        else:

            # Compute belief state
            self.pi_0, self.pi_1 = self.p_s_giv_o(self.o_t)

            if self.task_agent_analysis:
                p_d_0 = norm.cdf(0, self.o_t, self.sigma)
            else:

                if self.pi_0 >= self.pi_1:
                    p_d_0 = 1
                else:
                    p_d_0 = 0

        self.p_d_t = [p_d_0, 1-p_d_0]
        self.d_t = np.random.binomial(1, self.p_d_t[1])

    def eval_poly(self):
        """ This function evaluates the polynomial

        :return: poly_eval: Evaluated polynomial
        """

        poly_int = np.polyint(np.append(self.c_t, [0]))  # indefinite integral of polynomial
        poly_eval = np.polyval(poly_int, [0, 1])  # evaluate polynomial in [0, 1]
        poly_eval = np.diff(poly_eval)  # difference of evaluated polynomial

        return poly_eval

    def softmax(self, v_a_t):
        """ This function implements the softmax action selection

        :param v_a_t: Action values
        :return: p_a_t: Choice probabilites
        """

        p_a_t = np.exp(np.dot(v_a_t, self.beta)) / np.sum(np.exp(np.dot(v_a_t, self.beta)))

        return p_a_t

    def compute_valence(self, pi_0, pi_1):
        """ This function computes the action-dependent reward probability (action values)

        :param pi_0: belief in favor of s_t = 0
        :param pi_1: belief in favor of s_t = 1
        :return: v_a_t: Vector containing action values
        """

        if self.eval_ana:

            # Conditional expected value of \mu given o_(1:t-1), r_(1:t-1)
            self.E_mu_t = self.eval_poly()

        else:

            self.E_mu_t = np.dot(self.mu_for_ev, self.p_mu)
            self.G = np.dot(self.mu_for_ev, self.p_mu)

        # Action valence evaluation
        v_a_0 = (pi_0 - pi_1) * self.E_mu_t + pi_1
        v_a_1 = (pi_1 - pi_0) * self.E_mu_t + pi_0

        # Concatenate action valences
        v_a_t = [v_a_0, v_a_1]

        return v_a_t

    def get_q_s_a(self):
        """ This function computes q-values q_0_1, q_1_0 and q_1_1 based on q_0_0 of the Q-learning model

        :return: q_0_1, q_1_0, q_1_1
        """

        q_0_1 = 1-self.q_0_0
        q_1_0 = 1-self.q_0_0
        q_1_1 = self.q_0_0

        return q_0_1, q_1_0, q_1_1

    @staticmethod
    # def compute_capital_q(self, q_0_0, q_1_0, q_0_1, q_1_1, pi_0, pi_1):
    def compute_capital_q(q_0_0, q_1_0, q_0_1, q_1_1, pi_0, pi_1):
        """ This function computes Q-values Q_0 and Q_1 of the Q-learning model

        :param q_0_0: State 0, action 1
        :param q_1_0: State 1, action 0
        :param q_0_1: State 0, action 1
        :param q_1_1: State 1, action 1
        :param pi_0: Belief over state 0
        :param pi_1: Belief over state 1
        :return: capital_q_a: Q-values
        """
        # todo: state, action checken

        capital_q_0 = q_0_0 * pi_0 + q_1_0 * pi_1
        capital_q_1 = q_0_1 * pi_0 + q_1_1 * pi_1

        capital_q_a = [capital_q_0, capital_q_1]

        return capital_q_a

    def integrate_voi(self, voi, **kwargs):
        """ This function computes the integral of a variable of interest (voi) that is
            conditional on the contrast difference

        :param voi: Variable of interest (0: Action values, 1: q_0 and q_1, 3: q_0_0)
        :return: int_voi:  Integrated variable of interest
        """

        # Todo: überlegen ob q_0_0 etc und q_0 and q_1 nicht zu ähnlich klingen. eventuell neue namen!

        # If provided, get current reward for voi 2 (q_0)
        r_t = kwargs.get('r_t', None)

        # Initialize matrix that will be filled with variable of interest
        voi_matrix = np.full([len(self.set_o), 2], np.nan)

        # Evaluate distribution over observations
        # todo: sollte ich das alles nochmal mit sampling validieren, müsste ich hier also u anstatt o reinmachennn
        # Vorher mit dirk besprechen!
        p_o_giv_u = stats.norm.pdf(self.set_o, self.o_t, self.sigma)

        # Normalize evaluated probabilities
        p_o_giv_u_norm = p_o_giv_u / sum(p_o_giv_u)

        # Cycle over possible observations
        for i in range(0, len(self.set_o)):

            # Compute current belief state
            pi_0, pi_1 = self.p_s_giv_o(self.set_o[i])

            if voi == 0:

                # Action values

                # Integrate over observations to obtain expected values
                if self.agent == 1 or self.agent == 2 or self.agent == 3:
                    v_a_t = self.compute_valence(pi_0, pi_1)
                else:
                    q_0_1, q_1_0, q_1_1 = self.get_q_s_a()
                    v_a_t = self.compute_capital_q(self.q_0_0, q_1_0, q_1_0, q_1_1, pi_0, pi_1)
                voi_matrix[i, 0] = v_a_t[0]
                voi_matrix[i, 1] = v_a_t[1]

            elif voi == 1:

                # q_0 and q_1
                q_0, q_1 = self.compute_q(r_t, pi_0, pi_1)
                voi_matrix[i, 0] = q_0
                voi_matrix[i, 1] = q_1

            elif voi == 2:

                # q_0_0
                q_0_0 = self.compute_q_0_0(pi_0, pi_1)
                voi_matrix[i, 0] = q_0_0

        if not voi == 2:

            # Expected values and q_0, q_1
            q_0 = np.array([sum(voi_matrix[:, 0] * p_o_giv_u_norm)])
            q_1 = np.array([sum(voi_matrix[:, 1] * p_o_giv_u_norm)])
            int_voi = [q_0, q_1]

        else:

            # q_0_0
            q_0_0 = sum(voi_matrix[:, 0] * p_o_giv_u_norm)
            int_voi = q_0_0

        return int_voi

    def ev_cat(self):
        """ This function computes the action values for the categorical agents

        :return: v_a_t: Categorical action values
        """

        # Compute belief state categorically
        pi_0, pi_1 = self.cat_bs()

        if self.agent == 2 or self.agent == 3:

            # Compute action values
            v_a_t = self.compute_valence(pi_0, pi_1)

        elif self.agent == 5 or self.agent == 6:

            # Compute q_0_1 - q_1_1
            q_0_1, q_1_0, q_1_1 = self.get_q_s_a()

            # Compute Q-values
            capital_q_a = self.compute_capital_q(self.q_0_0, q_1_0, q_1_0, q_1_1, pi_0, pi_1)

            # Compute action values
            v_a_t = [np.array([capital_q_a[0]]), np.array([capital_q_a[1]])]

        else:

            # Provide warning if applied to wrong agent
            warnings.warn("v_a_t is not defined")
            v_a_t = np.nan

        return v_a_t

    def compute_mixture(self, first_comp, second_comp, q_learn=False):
        """ This function compute the mixtures between the agents

        :param first_comp: First mixture component
        :param second_comp: Second mixture component
        :param q_learn: If True, computes q_0_0
        :return: mixture: Mixture between components
        """

        if not q_learn:
            mixture_0 = first_comp[0] * self.lambda_param + second_comp[0] * (1 - self.lambda_param)
            mixture_1 = first_comp[1] * self.lambda_param + second_comp[1] * (1 - self.lambda_param)
            mixture = [mixture_0, mixture_1]
        else:
            mixture = first_comp * self.lambda_param + second_comp * (1-self.lambda_param)

        return mixture

    def decide_e(self):
        """ This function implements the agent's economic choice strategy

            If the agent is used for data analysis, we integrate over the range of observations
        """

        if self.agent == 0:

            # Determine expected values for random choices
            self.v_a_t = [0.5, 0.5]
            self.p_a_t = [0.5, 0.5]

        elif self.agent == 1:

            if self.task_agent_analysis:

                # Compute action values by integrating over observations
                voi = 0
                self.v_a_t = self.integrate_voi(voi)

            else:

                # Compute belief state
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute action valences
                self.v_a_t = self.compute_valence(pi_0, pi_1)

        elif self.agent == 2:

            # Compute categorical action values
            self.v_a_t = self.ev_cat()

        elif self.agent == 3:

            if self.task_agent_analysis:

                # Compute action values by integrating over observations
                voi = 0
                v_a_t_ag_1 = self.integrate_voi(voi)

                # Compute categorical action values
                v_a_t_ag_2 = self.ev_cat()

                # Compute action value of mixture between both models based on \lambda
                self.v_a_t = self.compute_mixture(v_a_t_ag_1, v_a_t_ag_2)

            else:

                # Compute belief state
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute action values
                v_a_t_ag_1 = self.compute_valence(pi_0, pi_1)

                # Compute categorical action values
                v_a_t_ag_2 = self.ev_cat()

                # Compute action value of mixture between both models based on \lambda
                self.v_a_t = self.compute_mixture(v_a_t_ag_1, v_a_t_ag_2)

        elif self.agent == 4:

            if self.task_agent_analysis:

                # Compute action values by integrating over observations
                voi = 0
                self.v_a_t = self.integrate_voi(voi)

            else:

                # Compute belief state
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute action values
                q_0_1, q_1_0, q_1_1 = self.get_q_s_a()
                self.v_a_t = self.compute_capital_q(self.q_0_0, q_1_0, q_1_0, q_1_1, pi_0, pi_1)

        elif self.agent == 5:

            # Compute categorical action values
            self.v_a_t = self.ev_cat()

        elif self.agent == 6:

            if self.task_agent_analysis:

                # Compute action values by integrating over observations
                voi = 0
                v_a_t_ag_4 = self.integrate_voi(voi)

                # Compute categorical action values
                v_a_t_ag_5 = self.ev_cat()

                # Compute action value of mixture between both models based on \lambda
                self.v_a_t = self.compute_mixture(v_a_t_ag_4, v_a_t_ag_5)
            else:
                # Compute belief state
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute action values
                q_0_1, q_1_0, q_1_1 = self.get_q_s_a()
                v_a_t_ag_4 = self.compute_capital_q(self.q_0_0, q_1_0, q_1_0, q_1_1, pi_0, pi_1)

                # Compute categorical action values
                v_a_t_ag_5 = self.ev_cat()

                # Compute action value of mixture between both models based on \lambda
                self.v_a_t = self.compute_mixture(v_a_t_ag_4, v_a_t_ag_5)

        # Compute choice probabilities
        if not self.agent == 0:
            self.p_a_t = self.softmax(self.v_a_t)

        # todo: unter umständen nur bei simulationen samplen
        # Sample agent action
        self.a_t = np.random.binomial(1, self.p_a_t[1])

    def compute_action_dep_rew(self, r_t):
        """ This function computes the action dependent reward r_t

        :param r_t: Current absolute reward
        :return: r_t: Action  dependent reward
        """

        r_t = r_t + (self.a_t * ((-1) ** (2 + r_t)))

        return r_t

    def compute_q(self, r_t, pi_0, pi_1):
        """ This function computes q_0 and q_1

        :param r_t: Current reward
        :param pi_0: Belief in favor of s_t = 0
        :param pi_1: Belief in favor of s_t = 1
        :return: q_0, q_1: Computed variables
        """

        # Evaluate the degree of the resulting polynomial (ohne self?)
        self.t = self.c_t.size + 1

        # Evaluate action-dependent reward value
        self.r_t = self.compute_action_dep_rew(r_t)

        if self.eval_ana:
            # Evaluate the gamma factor
            self.G = self.eval_poly()

            # Evaluate common denominator of q_0 and q_1
            self.C = (pi_0 - pi_1) * ((1 - self.G) ** (1 - self.r_t)) * \
                     (self.G ** self.r_t) + pi_1

            # Evaluate q_0 and q_1
            q_0 = (pi_1 ** self.r_t) * (pi_0 ** (1 - self.r_t)) / self.C
            q_1 = ((-1) ** (self.r_t + 1)) * (pi_0 - pi_1) / self.C

        else:
            # Evaluate q_0 and q_1
            self.q_0_num = (pi_1 ** self.r_t) * (pi_0 ** (1 - self.r_t))
            self.q_1_num = ((-1) ** (self.r_t + 1)) * (pi_0 - pi_1)

            self.product = self.product * (self.q_1_num * self.p_mu + self.q_0_num)

            self.mu = self.product
            self.mu_for_ev = self.mu / np.sum(self.mu)

            q_0 = self.q_0_num
            q_1 = self.q_1_num

        return q_0, q_1

    def compute_q_0_0(self, pi_0, pi_1):
        """ This function computes q_0_0

        :param pi_0: Belief over state 0
        :param pi_1: Belief over state 1
        :return: Computed q_0
        """

        if pi_0 >= pi_1:
            q_0_0 = self.q_0_0 + pi_0 * self.alpha * (self.r_t - self.q_0_0)
        else:
            q_0_0 = self.q_0_0 + pi_1 * self.alpha * ((1 - self.r_t) - self.q_0_0)

        return q_0_0

    def update_coefficients(self):
        # This function updates the polynomial coefficients

        # Initialize update coefficients
        self.d = np.zeros(self.t)

        # Evaluate last element of d_t
        self.d[-1] = self.q_0 * self.c_t[-1]

        # Evaluate d_1,... (if existent)
        for n in range(0, (self.t - 2)):
            self.d[-n - 2] = self.q_1 * self.c_t[-n - 1] + self.q_0 * self.c_t[-n - 2]

        # Evaluate first element of d_t
        self.d[0] = self.q_1 * self.c_t[0]

        # Update the coefficients
        self.c_t = self.d

    def q_a2(self, r_t):
        """ This function computes the q-values of agent A2

        :param r_t: Current reward
        :return: q_s: Computed q-values
        """

        # Compute belief state based on perceptual decision
        pi_0, pi_1 = self.cat_bs()

        # Compute q based on perceptual decision
        q_0, q_1 = self.compute_q(r_t, pi_0, pi_1)
        q_s = [q_0, q_1]

        return q_s

    def update_a4(self, r_t):
        """ This function computes q_0_0 of Agent A4

        :param r_t: Current reward
        :return: Computed q_0_0
        """

        # Evaluate action-dependent reward value
        self.r_t = self.compute_action_dep_rew(r_t)

        # Compute q_0_0 by integrating over range of possible observations
        voi = 2
        q_0_0 = self.integrate_voi(voi)

        return q_0_0

    def update_a5(self, r_t):
        """ This function computes q_0_0 of agent A5

        :param r_t: Current reward
        :return: Computed q_0_0
        """

        # Compute belief state based on perceptual decision
        pi_0, pi_1 = self.cat_bs()

        # Evaluate action-dependent reward value
        self.r_t = self.compute_action_dep_rew(r_t)

        # Compute q_0_0 categorically
        q_0_0 = self.compute_q_0_0(pi_0, pi_1)

        return q_0_0

    def learn(self, r_t):
        """ This function implements the agent's learning process

            If the agent is used for data analysis, we integrate over the range of observations

        :param r_t: Current reward
        """

        if self.agent == 0:

            # Random choice agent (A0) constantly represents p(\mu) = 0.5
            self.G = 0.5

        elif self.agent == 1:

            if self.task_agent_analysis:

                # Compute q's by integrating over possible observations
                voi = 1
                q_s = self.integrate_voi(voi, r_t=r_t)
                self.q_0 = q_s[0]
                self.q_1 = q_s[1]

                # Update polynomial coefficients
                self.update_coefficients()

            else:
                # Compute belief state using sampled observation
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute q using sampled observation
                self.q_0, self.q_1 = self.compute_q(r_t, pi_0, pi_1)

                # Update polynomial coefficients
                self.update_coefficients()

        elif self.agent == 2:

            # Compute q_0 and q_1
            q_s = self.q_a2(r_t)
            self.q_0 = q_s[0]
            self.q_1 = q_s[1]

            # Update polynomial coefficients
            self.update_coefficients()

        elif self.agent == 3:

            # todo: muss man hier nicht auch action-dependent reward berechnen?

            if self.task_agent_analysis:

                # Compute q's by integrating over possible observations
                voi = 1
                q_s_ag_1 = self.integrate_voi(voi, r_t=r_t)

                # Compute q_0 and q_1 categorically
                q_s_ag_2 = self.q_a2(r_t)

                # Compute final q as a function of \lambda
                q = self.compute_mixture(q_s_ag_1, q_s_ag_2)
                self.q_0 = q[0]
                self.q_1 = q[1]

                # Update polynomial coefficients
                self.update_coefficients()

            else:

                # Compute belief state using sampled observation
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute q using sampled observation
                q_0_ag_1, q_1_ag_1 = self.compute_q(r_t, pi_0, pi_1)
                q_s_ag_1 = [q_0_ag_1, q_1_ag_1]

                # Compute q categorically
                q_s_ag_2 = self.q_a2(r_t)

                # Compute final q as a function of \lambda
                q = self.compute_mixture(q_s_ag_1, q_s_ag_2)
                self.q_0 = q[0]
                self.q_1 = q[1]

                # Update polynomial coefficients
                self.update_coefficients()

        elif self.agent == 4:

            if self.task_agent_analysis:

                # Compute q_0_0 by integrating over possible observations
                self.q_0_0 = self.update_a4(r_t)

                # todo: das noch klären
                self.E_mu_t = self.q_0_0
                self.G = self.E_mu_t

            else:

                # Compute belief state using sampled observation
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute action-dependent reward
                self.r_t = self.compute_action_dep_rew(r_t)

                # Compute q based on sampled observation
                self.q_0_0 = self.compute_q_0_0(pi_0, pi_1)

                # todo: das noch klären
                self.E_mu_t = self.q_0_0
                self.G = self.E_mu_t

        elif self.agent == 5:

            # Compute q_0_0 categorically
            self.q_0_0 = self.update_a5(r_t)

            # todo: das noch klären
            self.E_mu_t = self.q_0_0
            self.G = self.E_mu_t

        elif self.agent == 6:

            if self.task_agent_analysis:

                # Compute q_0_0 by integrating over possible observations
                q_0_0_ag_4 = self.update_a4(r_t)

                # Compute q_0_0 categorically
                q_0_0_ag_5 = self.update_a5(r_t)

                # Compute final q_0_0 as a function of \lambda
                self.q_0_0 = self.compute_mixture(q_0_0_ag_4, q_0_0_ag_5, q_learn=True)

                # todo: das noch klären
                self.E_mu_t = self.q_0_0
                self.G = self.E_mu_t

            else:

                # Compute belief state using sampled observation
                pi_0, pi_1 = self.p_s_giv_o(self.o_t)

                # Compute relative reward
                self.r_t = self.compute_action_dep_rew(r_t)

                # Compute q_0_0 by integrating over possible observations
                q_0_0_ag_4 = self.compute_q_0_0(pi_0, pi_1)

                # Compute q_0_0 categorically
                q_0_0_ag_5 = self.update_a5(r_t)

                # Compute final q_0_0 as a function of \lambda
                self.q_0_0 = self.compute_mixture(q_0_0_ag_4, q_0_0_ag_5, q_learn=True)

                # todo: das noch klären
                self.E_mu_t = self.q_0_0
                self.G = self.E_mu_t
