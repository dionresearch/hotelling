Hotelling
=========


![image](https://github.com/dionresearch/hotelling/raw/master/png/hotelling_logo.png)


Status
------
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/v/hotelling.svg)](https://pypi.python.org/pypi/hotelling) 
![Dev](https://github.com/dionresearch/hotelling/actions/workflows/dev.yml/badge.svg)
![Release](https://github.com/dionresearch/hotelling/actions/workflows/release.yml/badge.svg)

About
-----

Hotelling implements one and two sample Hotelling T\^2 (T-squared) tests.
It also implements Hotelling Control Charts (Multivariate) and multiple
Univariate Control Charts

  ![image](https://github.com/dionresearch/hotelling/raw/master/png/hotelling_control_chart.png)

  ![image](https://github.com/dionresearch/hotelling/raw/master/png/univariate_chart.png)


-   Free software: MIT license
-   Documentation: <https://dionresearch.github.io/hotelling/>.

Features
--------

-   Stats module covering hotelling t^2 (t-squared) statistics, f-value and p-value
-   plots module covering Univariate Control Chart and Hotelling Control Chart
-   with the optional `dask` (and `distributed`) module, can handle large datasets efficiently
-   with the optional `plotly` module, provides interactive charts:

  ![image](https://github.com/dionresearch/hotelling/raw/master/png/interactive.png)


For this to work properly, you have to install `plotly` version 0.5 or greater. This is available from pypi, or through
the plotly channel for conda: `conda install plotly -c plotly`

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
