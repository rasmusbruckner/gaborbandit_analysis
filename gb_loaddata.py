import numpy as np
import pandas as pd


def loaddata(f_names):
    """ This function loads Gabor-bandit task data and check if they are complete

    :param f_names:     List with all file names
    :return:all_data:   Data frame that contains all data
    """

    # Initialize arrays
    T_exp1 = np.full(len(f_names), np.nan)  # number of trials in perceptual decision making task
    T_exp2 = np.full(len(f_names), np.nan)  # number of trials in economic decision making task
    T_exp3 = np.full(len(f_names), np.nan)  # number of trials in main task

    # Put data in data frame
    for i in range(0, len(f_names)):

        if i == 0:
            # Load data of participant 0
            all_data = pd.read_csv(f_names[0])
            new_data = all_data
        else:
            # Load data of participant 1,..,N
            new_data = pd.read_csv(f_names[i])

        # Count number of respective trials
        T_exp1[i] = np.sum((new_data['whichLoop'] == 'patches'))
        T_exp2[i] = np.sum(new_data['whichLoop'] == 'main_safe')
        T_exp3[i] = np.sum(new_data['whichLoop'] == 'main_unc')

        # Check if data are complete
        if T_exp1[i] == 100 and T_exp2[i] >= 150 and T_exp3[i] >= 300:
            new_data['complete'] = True
        else:
            new_data['complete'] = False

        # Add ID that indicates participant order
        new_data['id'] = i

        # Append data frame
        if i > 0:
            all_data = all_data.append(new_data, ignore_index=True)

    return all_data
