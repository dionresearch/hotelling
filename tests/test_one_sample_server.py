#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `hotelling` package."""

import numpy as np
import pytest
import pandas as pd

from hotelling.helpers import load_df
from hotelling.stats import hotelling_t2


try:
    import distributed

    def test_hotelling_test_csv_one_sample_server():
        x = load_df('data/shoes.csv', server="localhost", index_col='Subject')
        res = hotelling_t2(x, np.asarray([7, 8, 5, 7, 9]))
        assert round(res[0], 4) == 52.6724  # T2
        assert round(res[1], 4) == 8.7787  # F
        assert round(res[2], 5) == 0.00016  # P-value

except ModuleNotFoundError:
    @pytest.mark.skip(reason="distributed module is not available")
    def test_hotelling_test_csv_one_sample_server():
        pass