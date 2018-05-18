import numpy as np


def gb_invcumcoeff(y, c_F):
    """ This function evaluates inverse cumulative density function F^(-1) described by polynomial coefficients
        by evaluating the root of the polynomial

        We apply inverse transform sampling, that is, we take uniform samples \in [0, 1],
        compute the value F(x) = y and get a random number drawn fro)m f(x)

    :param y: Realization of a uniform RV on [0,1]
    :param c_F: 1 x t+1 array of polynomial coefficients of F(x)
    :return: x: Realization of a RV with pdf f(x) = F'(x) on [0,1]
    """

    C = c_F
    C[-1] = (C[-1] - y)

    # Return roots of polynomial
    R = np.roots(C)

    # Extract real part of roots on [0, 1]
    x = np.real(R[np.imag(R) == 0])
    x = x[(x >= 0) & (x <= 1)]

    return x
