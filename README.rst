Hotelling
=========

.. image:: https://github.com/dionresearch/hotelling/raw/master/png/hotelling_logo.png

Status
------
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
        :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/v/hotelling.svg
        :target: https://pypi.python.org/pypi/hotelling
.. image:: https://github.com/dionresearch/hotelling/actions/workflows/dev.yml/badge.svg
.. image:: https://github.com/dionresearch/hotelling/actions/workflows/release.yml/badge.svg

About
-----
Hotelling implements one and two sample Hotelling T^2 (T-squared) tests.
It also implements Hotelling Control Charts (Multivariate) and multiple
Univariate Control Charts

.. figure:: https://github.com/dionresearch/hotelling/raw/master/png/hotelling_control_chart.png
   :alt: Hotelling Control Chart

   Hotelling Control Chart

.. figure:: https://github.com/dionresearch/hotelling/raw/master/png/univariate_chart.png
   :alt: Univariate Control Chart

   Univariate Control Chart

-  Free software: MIT license
-  Documentation: https://dionresearch.github.io/hotelling/
-  Extra data in `tests/data`

Features
--------

-  Stats module covering hotelling t^2 (t-squared) statistics, f-value
   and p-value
-  plots module covering Univariate Control Chart and Hotelling Control
   Chart
-  with the optional `dask` (and `distributed`) module, can handle
   large datasets efficiently
-  with the optional `plotly` module, provides interactive charts:

.. figure:: https://github.com/dionresearch/hotelling/raw/master/png/interactive.png
   :alt: Interactive Control Chart


For this to work properly, you have to install `plotly` version 0.5 or greater. This is available from pypi, or through
the plotly channel for conda: `conda install plotly -c plotly`

Credits
-------

This package was created with
`Cookiecutter <https://github.com/audreyr/cookiecutter>`__ and the
`audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`__
project template.

.. |image| image:: https://img.shields.io/pypi/v/hotelling.svg
   :target: https://pypi.python.org/pypi/hotelling
.. |Documentation Status| image:: https://readthedocs.org/projects/hotelling/badge/?version=latest
   :target: https://hotelling.readthedocs.io/en/latest/?badge=latest
