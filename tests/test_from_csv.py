#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hotelling` package."""

import numpy as np
import pytest
import pandas as pd

from hotelling.stats import hotelling_t2


def test_hotelling_test_csv_one_sample():
    x = pd.read_csv('data/shoes.csv', index_col=0)
    res = hotelling_t2(x, np.asarray([7, 8, 5, 7, 9]))
    assert round(res[0], 4) == 52.6724  # T2
    assert round(res[1], 4) == 8.7787  # F
    assert round(res[2], 5) == 0.00016  # P-value

