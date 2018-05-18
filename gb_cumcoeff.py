import numpy as np


def gb_cumcoeff(c_f):
    """ This function returns the polynomial coefficients of the cumulative density function of a probability
        density function on [0,1], which is expressed in terms of the coefficients of a t-degree polynomial

    Formally, if

            f(x) = 1_[0,1](x) \sum_(k=0)^ c_k x^k

    then
            F(x) = 0                                x \in -\infty 0[
            F(x) \sum_(k=0)^ c_k\(k+1) x^(k+1)      x \in [0,1]
            F(x) = 1

    Note that F(x) is a (t+1)-degree polynomial w

            F(x) = \sum_(k=0)^(t+1) d_k x^k

    with coefficient d_0 := 0 and

            d_k = c_(k-1)/k

    :param c_f: 1 x t array of polynomial coefficients of f(x)
    :return: c_F: 1 x t+1 array of polynomial coefficients of F(x)
    """

    # Evaluate polynomial order of input coefficients
    t = len(c_f)

    # Transform coefficients
    x = np.flip(np.arange(1, t+1), 0)
    c_F = np.append(c_f/x, [0])

    return c_F
