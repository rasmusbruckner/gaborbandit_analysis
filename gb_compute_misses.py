import numpy as np

# check if this can be part of gb_utilities


def compute_misses(n_misses, n_subj):
    """ This function computes descriptive statistics of missed trials

    :param n_misses: Number of missed trials
    :param n_subj: Number of participants
    :return: n_misses_min: Minimum of misses
             n_misses_max: Maximum of misses
             n_misses_mean: Mean of misses
             n_misses_sem: Standard error of the mean
    """

    # Compute number of missed trials
    n_misses_min = np.min(n_misses)
    n_misses_max = np.max(n_misses)
    n_misses_mean = np.mean(n_misses)
    n_misses_sd = np.std(n_misses)
    n_misses_sem = n_misses_sd/np.sqrt(n_subj)

    return n_misses_min, n_misses_max, n_misses_mean, n_misses_sem
