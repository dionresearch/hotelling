#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `hotelling` package."""

import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from hotelling.stats import hotelling_t2


def test_hotelling_test_array_two_sample():
    x = np.asarray([[23, 45, 15], [40, 85, 18], [215, 307, 60], [110, 110, 50], [65, 105, 24]])
    y = np.asarray([[277, 230, 63], [153, 80, 29], [306, 440, 105], [252, 350, 175], [143, 205, 42]])
    res = hotelling_t2(x, y)
    assert round(res[0], 4) == 11.1037  # T2
    assert round(res[1], 4) == 2.7759  # F
    assert round(res[2], 5) == 0.15004  # p value


def test_hotelling_test_df_two_sample():
    x = pd.DataFrame([[23, 45, 15], [40, 85, 18], [215, 307, 60], [110, 110, 50], [65, 105, 24]])
    y = pd.DataFrame([[277, 230, 63], [153, 80, 29], [306, 440, 105], [252, 350, 175], [143, 205, 42]])
    res = hotelling_t2(x, y)
    assert round(res[0], 4) == 11.1037  # T2
    assert round(res[1], 4) == 2.7759  # F
    assert round(res[2], 5) == 0.15004  # p value


def test_hotelling_test_df_two_sample_no_bessel():
    x = pd.DataFrame([[23, 45, 15], [40, 85, 18], [215, 307, 60], [110, 110, 50], [65, 105, 24]])
    y = pd.DataFrame([[277, 230, 63], [153, 80, 29], [306, 440, 105], [252, 350, 175], [143, 205, 42]])
    res = hotelling_t2(x, y, bessel=False)
    assert round(res[0], 4) == 11.1037  # T2
    assert round(res[1], 4) == 2.2207  # F
    assert round(res[2], 5) == 0.17337


def test_nutrients_data_integrity_means_procedure():
    df = pd.read_csv('data/nutrient.txt', delimiter=' ',  skipinitialspace=True, index_col=0)
    res = df.describe().T
    assert (res['count'] == [737, 737, 737, 737, 737]).all()
    # mean
    assert_series_equal(res['mean'],
                        pd.Series([624.0492537, 11.1298996, 65.8034410, 839.6353460, 78.9284464],
                                  name='mean',
                                  index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                        check_less_precise=7)
    # for the next one, SAS displays 1633.54 for 'a' - that is an error, inconsistent. everything else is 7 digits
    # standard deviation
    assert_series_equal(res['std'],
                        pd.Series([397.2775401, 5.9841905, 30.5757564, 1633.5398283, 73.5952721],
                                  name='std',
                                  index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                        check_less_precise=7)
    # min
    assert_series_equal(res['min'],
                        pd.Series([7.4400000, 0, 0, 0, 0],
                                  name='min',
                                  index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                        check_less_precise=7)
    # max
    assert_series_equal(res['max'],
                        pd.Series([2866.44, 58.6680000, 251.0120000, 34434.27, 433.3390000],
                                  name='max',
                                  index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                        check_less_precise=7)


def test_nutrient_data_corr_procedure():
    df = pd.read_csv('data/nutrient.txt', delimiter=' ', skipinitialspace=True, index_col=0)
    # Covariance matrix
    cov = df.cov()
    assert_frame_equal(cov,
                       pd.DataFrame([[157829.444, 940.089, 6075.816, 102411.127, 6701.616],
                                     [940.089, 35.811, 114.058, 2383.153, 137.672],
                                     [6075.816, 114.058, 934.877, 7330.052, 477.200],
                                     [102411.127, 2383.153, 7330.052, 2668452.371, 22063.249],
                                     [6701.616, 137.672, 477.200, 22063.249, 5416.264]
                                     ],
                                    index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object'),
                                    columns=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                       check_less_precise=3)

    # Pearson Correlation
    corr = df.corr()
    assert_frame_equal(corr,
                       pd.DataFrame([[1.00000, 0.39543, 0.50019, 0.15781, 0.22921],
                                     [0.39543, 1.00000, 0.62337, 0.24379, 0.31260],
                                     [0.50019, 0.62337, 1.00000, 0.14676, 0.21207],
                                     [0.15781, 0.24379, 0.14676, 1.00000, 0.18352],
                                     [0.22921, 0.31260, 0.21207, 0.18352, 1.00000]
                                     ],
                                    index=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object'),
                                    columns=pd.Index(['calcium', 'iron', 'protein', 'a', 'c'], dtype='object')),
                       check_less_precise=3)


def test_mu0():
    """test_mu0

    One sample T-squared test. Hypothesis tested: mu = mu0

    :return:
    """
    # mu0 is the USDA recommended daily intake
    mu0 = np.array([1000, 15, 60, 800, 75])

    # 1985 USDA data
    df = pd.read_csv('data/nutrient.txt', delimiter=' ', skipinitialspace=True, index_col=0)

    res = hotelling_t2(df, mu0)
    assert round(res[0], 4) == 1758.5413  # T2
    assert round(res[1], 4) == 349.7968  # F
    assert round(res[2], 4) == 0.0000  # p-value
