# -*- coding: utf-8 -*-
"""Console script for hotelling."""

import sys
import click
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from hotelling.stats import hotelling_dict
from hotelling.plots import control_chart
from hotelling.helpers import savefig

matplotlib.rcParams["backend"] = "Agg"


@click.command()
@click.option("--x", prompt="X filename", help="Dataset X filename.")
@click.option("--y", help="Dataset Y filename.")
@click.option("--chart", help="Display control chart.", default=False)
@click.option("--output", help="filename or stdout")
def main(x, y=None, chart=None, output=None):
    """Console script for hotelling."""
    df1 = pd.read_csv(x)
    if y is None:
        df2 = None
    else:
        df2 = pd.read_csv(y)
    if chart:
        ax = control_chart(df1)
        if output == "stdout":
            savefig(plt)
        else:
            ax.get_figure().savefig(output)
    else:
        res = hotelling_dict(df1, df2)
        print(res)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
