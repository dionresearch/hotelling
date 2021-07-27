#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hotelling` package."""

import numpy as np

from hotelling.helpers import load_df
from hotelling.stats import hotelling_t2


def test_hotelling_test_csv_one_sample_no_dask():
    x = load_df('data/shoes.csv', index_col='Subject')
    res = hotelling_t2(x, np.asarray([7, 8, 5, 7, 9]))
    assert round(res[0], 4) == 52.6724  # T2
    assert round(res[1], 4) == 8.7787  # F
    assert round(res[2], 5) == 0.00016  # P-value


def test_hotelling_test_dat_one_sample_no_dask():
    data = load_df('data/sweat.dat')
    res = hotelling_t2(data, np.asarray([4, 50, 10]))
    assert round(res[0], 4) == 9.7388  # T2
    assert round(res[1], 4) == 2.9045  # F
    assert round(res[2], 5) == 0.06493  # P-value


def test_hotelling_test_dat_one_sample_no_mu_no_dask():
    data = load_df('data/sweat.dat')
    res = hotelling_t2(data)
    assert round(res[0], 4) == 1562.9873  # T2
    assert round(res[1], 4) == 466.1541  # F
    assert round(res[2], 5) == 0.0  # P-value


def test_hotelling_test_csv_one_sample_dask():
    x = load_df('data/shoes.csv', dask=True, index_col='Subject')
    res = hotelling_t2(x, np.asarray([7, 8, 5, 7, 9]))
    assert round(res[0], 4) == 52.6724  # T2
    assert round(res[1], 4) == 8.7787  # F
    assert round(res[2], 5) == 0.00016  # P-value


def test_hotelling_test_dat_one_sample_dask():
    data = load_df('data/sweat.dat', dask=True)
    res = hotelling_t2(data, np.asarray([4, 50, 10]))
    assert round(res[0], 4) == 9.7388  # T2
    assert round(res[1], 4) == 2.9045  # F
    assert round(res[2], 5) == 0.06493  # P-value
