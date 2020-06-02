# -*- coding: utf-8 -*-

"""Console script for hotelling."""
import sys
import click
import pandas as pd

from hotelling.stats import hotelling_dict


@click.command()
@click.option("--x", prompt="X filename", help="Dataset X filename.")
@click.option("--y", help="Dataset Y filename.")
def main(x, y=None):
    """Console script for hotelling."""
    df1 = pd.read_csv(x)
    if y is None:
        df2 = None
    else:
        df2 = pd.read_csv(y)
    res = hotelling_dict(df1, df2)
    print(res)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
