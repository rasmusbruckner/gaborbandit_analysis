import numpy as np
import pandas as pd
from GbTask import Task
from GbAgent import Agent


def gb_task_agent_int(task_vars, agent_vars, sim_vars, **kwargs):
    """ This function simulates the interaction of the task- and agent-object

    :param task_vars: Task parameters
    :param agent_vars: Agent parameters
    :param sim_vars: Simulation parameters
    :return: df: Data-frame with task-agent interaction results
    """

    real_outc = kwargs.get('real_outc', None)

    # Extract task variables
    T = task_vars.T

    # Initialize data frame for recorded variables
    df = pd.DataFrame(index=range(0, T), dtype='float')

    # Initialize variables
    s = np.full(T, np.nan)  # state
    u = np.full(T, np.nan)  # observation (objective)
    o = np.full(T, np.nan)  # observation (subjective)
    d = np.full(T, np.nan)  # perceptual decision
    e_mu_t = np.full(T, np.nan)  # expected value
    a = np.full(T, np.nan)  # economic decision
    p_a0_t = np.full(T, np.nan)  # choice probability a = 0
    p_a1_t = np.full(T, np.nan)  # choice probability a = 1
    p_a_t = np.full(T, np.nan)  # choice probability array
    ll = np.full(T, np.nan)  # log choice probability
    v_a_0 = np.full(T, np.nan)  # choice probability a = 0
    v_a_1 = np.full(T, np.nan)  # choice probability a = 1
    r = np.full(T, np.nan)  # reward
    pi_1 = np.full(T, np.nan)  # belief over being in s_t = 1
    block = np.full(T, np.nan)  # block
    corr = np.full(T, np.nan)  # correct economic decision
    trial = np.full(T, np.nan)  # trial

    # Task and agent class instantiation
    task = Task(task_vars)
    agent = Agent(agent_vars)

    # Cycle over trials t = 1,...T
    # ----------------------------
    for t in range(0, T):

        # Task-agent interaction
        # ----------------------
        if real_outc is None:
            task.state_sample()
            task.contrast_sample()
        else:
            task.s_t = real_outc['s_t'][t]
            task.c_t = real_outc['u_t'][t]

        # Contrast difference
        agent.observation_sample(task.c_t)

        # Perceptual decision making
        if sim_vars.take_pd == 1:
            agent.d_t = np.int(real_outc['d_t'][t])
        else:
            agent.decide_p()

        if task_vars.experiment == 2 or task_vars.experiment == 3:

            # Economic decision making
            agent.decide_e()

            # Reward delivery
            task.reward_sample(agent.a_t)

            # Learning
            agent.learn(task.r_t)

            # Determine if high reward action was chosen
            if (task.s_t == 0 and agent.a_t == 0) or (task.s_t == 1 and agent.a_t == 1):
                corr[t] = 1
            else:
                corr[t] = 0

        # Record model variables
        s[t] = task.s_t
        u[t] = task.c_t
        o[t] = agent.o_t
        pi_1[t] = agent.pi_1
        d[t] = agent.d_t
        if task_vars.experiment == 2 or task_vars.experiment == 3:
            e_mu_t[t] = agent.G
            v_a_0[t] = agent.v_a_t[0]
            v_a_1[t] = agent.v_a_t[1]
            a[t] = agent.a_t
            p_a0_t[t] = agent.p_a_t[0]
            p_a1_t[t] = agent.p_a_t[1]
            p_a_t[t] = agent.p_a_t[np.int(a[t])]
            ll[t] = np.log(agent.p_a_t[np.int(a[t])])
            r[t] = task.r_t
        trial[t] = t
        block[t] = sim_vars.block

    # Attach model variables to data frame
    df['s_t'] = s
    df['u_t'] = u
    df['o_t'] = o
    df['pi_1'] = pi_1
    df['d_t'] = d
    df['t'] = trial
    df['block'] = block
    df['corr'] = corr
    if task_vars.experiment == 2 or task_vars.experiment == 3:
        df['e_mu_t'] = e_mu_t
        df['v_a_0'] = v_a_0
        df['v_a_1'] = v_a_1
        df['a_t'] = a
        df['p_a0_t'] = p_a0_t
        df['p_a1_t'] = p_a1_t
        df['p_a_t'] = p_a_t
        df['ll'] = ll
        df['r_t'] = r

    return df
