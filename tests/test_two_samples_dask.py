#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hotelling` package."""

import numpy as np
import pytest
import pandas as pd

from hotelling.helpers import load_df
from hotelling.stats import hotelling_t2


def test_hotelling_test_csv_two_sample_no_dask():
    x = load_df('data/swiss_real.csv')
    y = load_df('data/swiss_fake.csv')
    res = hotelling_t2(x, y)
    assert round(res[0], 4) == 2412.4507  # T2
    assert round(res[1], 4) == 391.9217  # F
    assert round(res[2], 5) == 0.0  # P-value


def test_hotelling_test_csv_two_sample_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    y = load_df('data/swiss_fake.csv', dask=True)
    res = hotelling_t2(x, y)
    assert round(res[0], 4) == 2412.4507  # T2
    assert round(res[1], 4) == 391.9217  # F
    assert round(res[2], 5) == 0.0  # P-value
