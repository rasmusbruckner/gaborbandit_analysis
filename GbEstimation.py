import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from GbAgentVars import AgentVars
from GbAgent import Agent


class GbEstimation:
    # This class specifies instance variables and methods of the estimation object for parameter estimation

    def __init__(self, est_vars):
        """ This function defines the instance variables unique to each instance

            See GbEstVars.py
        """

        self.T = est_vars.T
        self.B = est_vars.B
        self.n_sim = est_vars.n_sim
        self.n_sp = est_vars.n_sp
        self.experiment = est_vars.experiment
        self.n_ker = est_vars.n_ker
        self.agent = est_vars.agent
        self.s_bnds = est_vars.s_bnds
        self.b_bnds = est_vars.b_bnds
        self.l_bnds = est_vars.l_bnds
        self.s_fixedsp = est_vars.s_fixedsp
        self.b_fixedsp = est_vars.b_fixedsp
        self.l_fixedsp = est_vars.l_fixedsp
        self.rand_sp = est_vars.rand_sp
        self.id = est_vars.id

    def d_llh(self, sigma, df, agent_vars):
        """ This function computes the likelihood of perceptual decisions

            Used to estimate the sigma parameter and to evaluate the likelihood with
            a fixed sigma parameter

        :param sigma: Perceptual sensitivity parameter
        :param df: Data frame that contains current data set
        :param agent_vars: Variables to initialize the agent object
        :return: llh: Cumulated negative log likelihood
        """

        # Number of trials
        n_trials = len(df)

        # Agent object initialization
        agent = Agent(agent_vars)

        # Set sensory noise parameter
        agent.sigma = sigma

        # Initialize likelihood array
        llh_d = np.full(n_trials, np.nan)

        # Cycle over trials
        for t in range(0, n_trials):

            # Set current observation
            agent.o_t = df['u_t'][t]

            # Evaluate probability of current perceptual decision
            agent.decide_p()

            # Compute log likelihood of perceptual decision
            llh_d[t] = np.log(agent.pi_t[np.int(df['d_t'][t])])

        # Sum negative log likelihoods
        llh = -1*np.sum(llh_d)

        return llh

    def a_llh(self, x, exp_data, sub_params, agent_vars, est_params):
        """ This function computes the likelihood of economic decisions

            Used to estimate the beta parameter of the softmax choice rule and
            the lambda parameter of agent A3. The function is also used to evaluate
            the likelihood of the economic decisions based on fixed parameters

        :param x: Free parameter
        :param exp_data: Data frame that contains current data set
        :param sub_params:
        :param agent_vars: Variables to initialize the agent object
        :param est_params:
        :return: llh: Cumulated negative log likelihood
        """

        # Initialize likelihood array
        llh_a = np.full([self.T, self.B], np.nan)

        # Initialize figure, if requested
        if est_params == 0:
            plt.figure(figsize=(20, 10))

        # Cycle over blocks
        for b in range(0, self.B):

            # Extract required data
            u = exp_data[exp_data['blockNumber'] == b]['u_t']
            d = exp_data[exp_data['blockNumber'] == b]['d_t']
            a = exp_data[exp_data['blockNumber'] == b]['a_t'] - 1
            r = exp_data[exp_data['blockNumber'] == b]['r_t']
            corr = exp_data[exp_data['blockNumber'] == b]['decision2.corr']

            # Agent object initialization
            agent = Agent(agent_vars)

            # Assign current sigma parameter
            agent.sigma = sub_params['sigma']  # sub_params['sigma']

            # Assign current beta and (if required) lambda parameter
            if self.agent == 0 or self.agent == 1 or self.agent == 2:
                agent.beta = x
            else:
                agent.beta = sub_params['beta']
                agent.lambda_param = x

            # Initialize block specific variables
            cp = np.full(self.T, np.nan)  # choice probability
            ev = np.full(self.T, np.nan)  # expected value

            # Cycle over trials
            for t in range(0, self.T):

                # Set current observation
                agent.o_t = u[t]

                # Set perceptual decision
                agent.d_t = np.int(d[t])

                # Evaluate probability of economic decisions
                agent.decide_e()

                # Set economic decision
                agent.a_t = np.int(a[t])

                # Extract probability of current economic decision
                cp[t] = agent.p_a_t[np.int(a[t])]

                # Compute log likelihood of perceptual decision
                llh_a[t, b] = np.log(agent.p_a_t[np.int(a[t])])

                # Agent task parameter update
                agent.learn(np.int(r[t]))

                # Extract expected value
                ev[t] = agent.G

            # Plot likelihoods for all blocks and trials
            if est_params == 0:

                # Index to place likelihood subplot at the correct position
                if b < 6:
                    ll_indx = 1
                else:
                    ll_indx = 7

                # Index to place choices subplot at the correct position
                if b < 6:
                    c_indx = 7
                else:
                    c_indx = 13

                # Plot choice probability and expected value
                plt.subplot(4, 6, b+ll_indx)  # initialize subplot
                plt.plot(ev)  # expected value
                plt.plot(cp)  # choice probability
                plt.ylim([0, 1])

                # Plot choices
                plt.subplot(4, 6, b+c_indx)
                plt.plot(corr, 'o')

        # Sum negative log likelihoods
        llh = -1*np.sum(llh_a)

        if est_params == 0:

            # Save the plot
            savename = 'gb_figures/single_trial/%s_%s.png' % (self.id, self.agent)
            plt.savefig(savename)
            plt.close()

        return llh

    def compute_bic(self, llh, n_params):
        """ This function compute the Bayesian information criterion (BIC)

            See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage

        :param llh: Negative log likelihood
        :param n_params: Number of free parameters
        :return: bic
        """

        bic = (-1 * llh) - (n_params / 2) * np.log(self.T)

        return bic

    def model_exp1(self, exp_data):
        """ This function implements the modeling steps for Experiment 1

            Used for both, evaluation with fixed parameters and estimation with
            free parameters.

        :param exp_data: Data of experiment 1
        :return: results_list: Results of evaluation and estimation procedure
        """

        # Agent parameters in GbAgentVars.py object
        agent_vars = AgentVars()

        # Add trial as index
        exp_data.loc[:, 'trial'] = np.linspace(0, len(exp_data) - 1, len(exp_data))
        exp_data = exp_data.set_index('trial')

        # Initialize with unrealistically high likelihood
        min_llh = 1000

        # Cycle over starting points
        for r in range(0, self.n_sp):

            # Starting point
            if self.rand_sp:
                x0 = np.array((np.random.rand()*self.s_bnds[0][1])+self.s_bnds[0][0])
            else:
                x0 = self.s_fixedsp

            # Parameter boundary
            bnds = self.s_bnds

            # Estimate parameter
            res = minimize(self.d_llh, x0, args=(exp_data, agent_vars), method='L-BFGS-B', bounds=bnds)

            # Extract optimization output maximum likelihood
            x = res.x  # parameter estimate
            llh = res.fun  # maximized log likelihood

            # Check if cumulated negative log likelihood is lower than the previous
            # one and select the lowest
            if llh < min_llh:
                min_llh = llh
                min_x = x

        # Add ID to results to check if participant order is maintained during parallelization
        self.id = np.array(list(set(exp_data['id'])))

        # Compute BIC with free parameter
        bic_d = self.compute_bic(min_llh, 1)

        # Initialize and fill list with results
        results_list = list()
        results_list.append(np.float(min_llh))
        results_list.append(np.float(bic_d))
        results_list.append(np.float(min_x))
        results_list.append(np.float(self.id))

        return results_list

    def model_exp23(self, exp_data, est_vars, sub_params, est_params=1):
        """ This function implements the modeling steps for Experiment 2 and 3

            Used for both, evaluation with fixed parameters and estimation with
            free parameters.

        :param exp_data: Data of current experiment
        :param est_vars: Variables that are used for estimation and evaluation
        :param sub_params: Participants' parameter estimates
        :param est_params: Indicates if we evaluate model or additionally estimate a free parameter
        :return: results: Results of evaluation and estimation procedure
        """

        # Agent parameters in GbAgentVars.py object
        agent_vars = AgentVars()

        # Select current agent
        agent_vars.agent = est_vars.agent
        self.agent = est_vars.agent

        exp_data.loc[:, 'trial'] = np.linspace(0, len(exp_data)-1, len(exp_data))
        exp_data = exp_data.set_index('trial')

        # Evaluate likelihood of perceptual decisions
        f_llh_d = self.d_llh(sub_params['sigma'], exp_data, agent_vars)

        exp_data.loc[:, 'trial'] = np.tile(np.linspace(0, self.T-1, self.T), self.B)
        exp_data = exp_data.set_index('trial')

        # Add ID to results to check if participant order is maintained during parallelization
        self.id = np.array(list(set(exp_data['id'])))

        # Estimate free parameters
        if est_params == 1:

            # Initialize with unrealistically high likelihood
            min_llh = 1000

            # Cycle over starting points
            for r in range(0, self.n_sp):

                # Estimate parameters
                if self.agent == 3:

                    # Starting point
                    if self.rand_sp:
                        x0 = np.random.rand()
                    else:
                        x0 = self.l_fixedsp

                    res = minimize(self.a_llh, x0, args=(exp_data, sub_params, agent_vars, est_params),
                                   method='L-BFGS-B', bounds=self.l_bnds)
                else:
                    # Starting point
                    if self.rand_sp:
                        x0 = np.random.rand() * self.b_bnds[0][1]
                    else:
                        x0 = self.b_fixedsp

                    res = minimize(self.a_llh, x0, args=(exp_data, sub_params, agent_vars, est_params),
                                   method='L-BFGS-B', bounds=self.b_bnds)

                # Extract maximum likelihood parameter estimate
                x = res.x

                # Extract maximized log likelihood
                llh = res.fun

                # Check if cumulated negative log likelihood is lower than the previous
                # one and select the lowest
                if llh < min_llh:
                    min_llh = llh
                    min_x = x

        # Evaluate experimental data
        else:

            # Evaluate the model
            if self.agent == 3:
                min_llh = self.a_llh(sub_params['lambda'], exp_data, sub_params, agent_vars, est_params)
            else:
                min_llh = self.a_llh(sub_params['beta'], exp_data, sub_params, agent_vars, est_params)

        # Compute BIC for economic decision
        bic_a = self.compute_bic(min_llh, est_params)

        # Compute BIC for perceptual decision without free parameters
        bic_d = self.compute_bic(f_llh_d, 0)

        # Initialize and fill list with results
        results_list = list()
        results_list.append(np.float(min_llh))
        results_list.append(np.float(bic_d))
        results_list.append(np.float(bic_a))
        results_list.append(np.float(self.id))
        results_list.append(np.float(self.agent))

        if est_params == 1:
            results_list.append(np.float(min_x))

        return results_list
