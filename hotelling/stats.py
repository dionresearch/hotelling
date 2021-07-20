# -*- coding: utf-8 -*-
"""Stats.py.

Hotelling's T-Squared multivariate test for one sample or two independent samples

See:
 Hotelling, Harold. (1931). The Generalization of Student's Ratio. Ann. Math. Statist. 2,
  no. 3, 360--378. doi:10.1214/aoms/1177732979.

https://projecteuclid.org/euclid.aoms/1177732979
"""
from warnings import warn

import numpy as np
from scipy.stats import f


def bessel_correction(x, y=None):
    """bessel_correction.

    Sampling tends to underestimate variability of a population. This is due to the fact that we are more likely to
    sample around the mean than near the extremities. Bessel's correction uses n−1 instead of n which is used to
    calculate variance etc, in order to correct for the bias in the estimation of the population variance.

    :param x: array-like, samples of observations
    :param y: array-like, samples of observations, optional
    :return: returns x_n - 1, y_n - 1
    """
    n1 = x.shape[0] - 1
    try:
        n1 = n1.compute()
    except AttributeError:
        pass
    if y is None:
        n2 = 0
    else:
        n2 = y.shape[0] - 1
        try:
            n2 = n2.compute()
        except AttributeError:
            pass
    return n1, n2


def inverse_covariance_matrix(x, y, bessel=True):
    """inverse_covariance_matrix.

    :param x: array-like, samples of observations
    :param y: array-like, samples of observations
    :param bessel: bool, apply bessel correction (default)
    :return: float, the pooled variance inverse, the pooled variance
    """
    _, *p = x.shape
    p = p[0] if p else 1
    s = pooled_covariance_matrix(x, y, bessel)
    try:
        ident_p = np.identity(p).compute
    except AttributeError:
        ident_p = np.identity(p)
    inv = np.linalg.solve(s, ident_p)
    return inv, s


def pooled_covariance_matrix(x, y, bessel=True):
    r"""pooled_covariance.

    Compute the pooled covariance matrix

    Equation:

    The pooled covariance matrix is defined as:

    .. math::
        S =  \\frac{n_xS_x + n_yS_y}{n_x+n_y}

    And with bessel correction as:

    .. math::
        S =  \\frac{(n_x-1)S_x + (n_y-1)S_y}{n_x+n_y-2}

    Reference
    ---------
    see: https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Pooled_covariance_matrix

    :param x: array-like, samples of observations
    :param y: array-like, samples of observations
    :param bessel: bool, apply bessel correction (default)
    :return: float, the pooled variance
    """
    if bessel:
        n1, n2 = bessel_correction(x, y)
    else:
        try:
            n1 = x.shape[0]
            n1 = n1.compute()
            n2 = y.shape[0]
            n2 = n2.compute()
        except AttributeError:
            n1 = x.shape[0]
            n2 = y.shape[0]
    try:
        s1 = n1 * x.cov().compute()

    except AttributeError:
        s1 = n1 * np.cov(x, rowvar=False)
    try:
        s2 = n2 * y.cov().compute()
    except AttributeError:
        s2 = n2 * np.cov(y, rowvar=False)
    s = (s1 + s2) / (n1 + n2)

    return s


def hotelling_t2(x, y=None, bessel=True, S=None):
    r"""hotelling_t2.

    Compute the Hotelling (T2) test statistic.

    It is the multivariate extension of the Student's t-test.
    Test the null hypothesis that two multivariate samples have the same underlying
    probability distribution, when specifying samples for x and y. The number of samples do not have
    to be the same, but the number of features does have to be equal.

    Equation:

    Hotelling's t-squared statistic is defined as:

    .. math::
        T^2 = n (\\bar{x} - {\mu})^{T} S^{-1} (\\bar{x} - {\mu})

    Where S is the pooled covariance matrix and ᵀ represents the transpose.

    The two sample t-squared statistic is defined as:

    .. math::
        T^2 = (\\bar{x} - \\bar{y})^{T} [S(\\frac1 n_x +\\frac 1 n_y)]^{-1} (\\bar{x}̄ - \\bar{y})

    References:
        - Hotelling, Harold. (1931). The Generalization of Student's Ratio. Ann. Math. Statist. 2, no. 3, 360--378.
          doi:10.1214/aoms/1177732979. https://projecteuclid.org/euclid.aoms/1177732979

        - Hotelling, Harold. (1955) Les Rapports entre les Methodes Statistiques recentes portant sur des Variables Multiples
          et l'Analyse Factorielle. 107-119.
          In: L'Analyse Factorielle et ses Applications. Centre National de la Recherche Scientifique, Paris.

        - Anderson T.W. (1992) Introduction to Hotelling (1931) The Generalization of Student’s Ratio.
          In: Kotz S., Johnson N.L. (eds) Breakthroughs in Statistics.
          Springer Series in Statistics (Perspectives in Statistics). Springer, New York, NY

    :param x: array-like, samples of observations for one or two sample test (required)
    :param y: for two sample test, array-like, samples of observations (optional), for one sample, list of means to test
    :param bessel: bool, apply bessel correction (default)
    :return:
        statistic: float,
            the t2 statistic
        f_value: float,
            the f value
        p_value: float,
            the p value
        s: 2d array,
            the pooled variance
    """  # noqa: W605
    try:
        nx, p = x.shape
    except AttributeError as ex:
        if "list" in str(ex):
            x = np.asarray(x)
            nx, *p = x.shape
            p = p[0] if p else 1
            y = np.asarray(y)
        else:
            warn("Error: The two samples must be in arrays or dataframes format.")
            raise ValueError

    # samples observed means
    try:
        nx = nx.compute()
        x_bar = x.mean(0).compute()
    except AttributeError:  # series has no attribute compute
        x_bar = x.mean(0)

    one_sample = False
    if y is None:
        # One sample T-squared
        one_sample = True
        y = np.zeros(p)
        ny = None
        py = p
        diff_bar = x_bar - y

    else:
        ny, *py = y.shape
        if len(py) == 0:
            one_sample = True
            py = p
            diff_bar = x_bar - y
        else:
            # Two sample T-squared
            py = py[0] if py else 1
            try:
                ny = ny.compute()
                y_bar = y.mean(0).compute()
            except AttributeError:  # series has no attribute compute
                y_bar = y.mean(0)
            # difference of means
            diff_bar = x_bar - y_bar
    if p != py:
        warn(
            f"Error: the two samples must have the same number of features ({p} != {py})."
        )
        raise ValueError

    # bessel correction ( -1 )
    if bessel:
        n1, n2 = bessel_correction(x, y)
    else:
        n1 = nx
        n2 = ny
    if one_sample:
        n = nx
    else:
        n = n1 + n2

    # calculate the T2 statistics
    # Technically, we use diff_bar.T for the transpose, but with Pandas, a 1 dimensional dataframe
    # is automatically aligned for @ and is not required
    if one_sample:
        if S is not None:
            cov = S
        else:
            try:
                cov = x.cov().compute()
            except AttributeError:
                try:
                    cov = x.cov()
                except AttributeError:
                    cov = np.cov(x, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        # for f test
        # term = (n - p) / (p * (n - 1))  # getting different results
        t2_stat = n * (diff_bar.T @ inv_cov @ diff_bar)
        if S is not None:
            return t2_stat
        # f statistic
        # TODO: use chi square instead of f statistic for large sample
        f_value = (n - p) * t2_stat / ((n - 1) * p)
    else:
        # pooled covariance
        inv_s, s = inverse_covariance_matrix(x, y, bessel)
        t2_stat = nx * ny / (nx + ny) * (diff_bar.T @ inv_s @ diff_bar)
        # f statistic
        # TODO: use chi square instead of f statistic for large sample
        f_value = (nx + ny - p - 1) * t2_stat / (n * p)

    # p-value
    p_value = f.sf(f_value, p, n - p)  # survival function, 1 - cdf

    # return the list of results
    return t2_stat, f_value, p_value, cov if one_sample else s


def hotelling_dict(x, y=None, bessel=True):
    """hotelling_dict.

    returns the same values as `hotelling_t2`, but in a dictionary - for API etc

    :param x: array-like, samples of observations for one or two sample test (required)
    :param y: for two sample test, array-like, samples of observations (optional), for one sample, list of means to test
    :return: dict
    """
    t2_stat, f_stat, p_value, s = hotelling_t2(x, y, bessel)
    return dict(t2_stat=t2_stat, f_stat=f_stat, p_value=p_value, pooled_var=s)
