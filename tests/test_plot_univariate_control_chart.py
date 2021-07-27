import matplotlib
matplotlib.rcParams["backend"] = "Agg"

import pytest
from hotelling.helpers import load_df
from hotelling.plots import univariate_control_chart


def drop_svg_date(filename):

    # These get new random id or are time dependent, so skip those lines
    exclude_list = [
        "<dc:date>",
        '" id="',
        "xlink:href=",
        " clip-path",
        "clipPath",
    ]
    svg = []
    with open(filename) as f:
        for line in f:
            if len([item for item in exclude_list if item in line]):
                continue
            svg.append(line)

    return svg


try:
    import plotly

    def test_univariate_plotly_no_dask():
        x = load_df('data/swiss_real.csv')
        figs = univariate_control_chart(x, legend_right=True, interactive=True, no_display=True)
        for i, fig in enumerate(figs):
            fig.write_html(f'plotly/test_univariate{i}_no_dask.html')

except ModuleNotFoundError:

    def test_univariate_plotly_no_dask():
        x = load_df('data/swiss_real.csv')
        with pytest.raises(NameError):
            fig = univariate_control_chart(x, legend_right=True, interactive=True)


def test_matplotlib_univariate_no_dask():
    x = load_df('data/swiss_real.csv')
    fig = univariate_control_chart(x, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_no_dask.svg')


def test_matplotlib_univariate_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    fig = univariate_control_chart(x, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_dask.svg')


def test_matplotlib_univariate_identical():
    svg1 = drop_svg_date('../svg/mpl_test_univariate_no_dask.svg')
    svg2 = drop_svg_date('../svg/mpl_test_univariate_dask.svg')
    # Line by line so we can get just the line that is different if it happens
    for line in range(len(svg1)):
        assert svg1[line] == svg2[line]


def test_matplotlib_univariate_cusum_no_dask():
    x = load_df('data/swiss_real.csv')
    fig = univariate_control_chart(x, cusum=True, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_cusum_no_dask.svg')


def test_matplotlib_univariate_cusum_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    fig = univariate_control_chart(x, cusum=True, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_cusum_dask.svg')


def test_matplotlib_univariate_cusum_identical():
    svg1 = drop_svg_date('../svg/mpl_test_univariate_cusum_no_dask.svg')
    svg2 = drop_svg_date('../svg/mpl_test_univariate_cusum_dask.svg')
    # Line by line so we can get just the line that is different if it happens
    for line in range(len(svg1)):
        assert svg1[line] == svg2[line]


def test_matplotlib_univariate_cusum_only_no_dask():
    x = load_df('data/swiss_real.csv')
    fig = univariate_control_chart(x, cusum=True, cusum_only=True, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_cusum_only_no_dask.svg')


def test_matplotlib_univariate_cusum_only_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    fig = univariate_control_chart(x, cusum=True, cusum_only=True, legend_right=True)
    fig.savefig('../svg/mpl_test_univariate_cusum_only_dask.svg')


def test_matplotlib_univariate_cusum_only_identical():
    svg1 = drop_svg_date('../svg/mpl_test_univariate_cusum_only_no_dask.svg')
    svg2 = drop_svg_date('../svg/mpl_test_univariate_cusum_only_dask.svg')
    # Line by line so we can get just the line that is different if it happens
    for line in range(len(svg1)):
        assert svg1[line] == svg2[line]
