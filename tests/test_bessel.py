import numpy as np
from hotelling.stats import bessel_correction


def test_bessel_x_y():
    x = np.asarray([7, 8, 5, 7, 9, 8])
    y = np.asarray([7, 8, 5, 7, 9])
    a, b = bessel_correction(x, y)
    assert a == 5
    assert b == 4


def test_bessel_x_no_y():
    x = np.asarray([7, 8, 5, 7, 9, 8])
    a, b = bessel_correction(x)
    assert a == 5
    assert b == 0


def test_bessel_x_y_none():
    x = np.asarray([7, 8, 5, 7, 9, 8])
    y = None
    a, b = bessel_correction(x, y)
    assert a == 5
    assert b == 0
