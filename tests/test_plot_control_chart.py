import pytest

from hotelling.helpers import load_df
from hotelling.plots import control_chart


def drop_svg_date(filename):

    # These get new random id or are time dependent, so skip those lines
    exclude_list = [
        "<dc:date>",
        " clip-path",
        "clipPath",
        "xlink:href=",
        '" id="'
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

    def test_plotly_no_dask():
        x = load_df('data/swiss_real.csv')
        fig = control_chart(x, alpha=0.01, legend_right=True, interactive=True, no_display=True)
        fig.write_html('plotly/test_no_dask.html')

except ModuleNotFoundError:

    def test_plotly_no_dask():
        x = load_df('data/swiss_real.csv')
        fig = control_chart(x, alpha=0.01, legend_right=True, interactive=True)
        with pytest.raises(AttributeError):
            fig.write_html('plotly/test_no_dask.html')


def test_matplotlib_no_dask():
    x = load_df('data/swiss_real.csv')
    ax = control_chart(x, alpha=0.01, legend_right=True)
    ax.get_figure().savefig('../svg/mpl_test_no_dask.svg')


def test_matplotlib_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    ax = control_chart(x, alpha=0.01, legend_right=True)
    ax.get_figure().savefig('../svg/mpl_test_dask.svg')


def test_matplotlib_identical():
    svg1 = drop_svg_date('../svg/mpl_test_no_dask.svg')
    svg2 = drop_svg_date('../svg/mpl_test_dask.svg')
    # Line by line so we can get just the line that is different if it happens
    for line in range(len(svg1)):
        assert svg1[line] == svg2[line]


def test_matplotlib_cusum_no_dask():
    x = load_df('data/swiss_real.csv')
    ax = control_chart(x, alpha=0.01, legend_right=True, cusum=True)
    ax.get_figure().savefig('../svg/mpl_test_cusum_no_dask.svg')


def test_matplotlib_cusum_dask():
    x = load_df('data/swiss_real.csv', dask=True)
    ax = control_chart(x, alpha=0.01, legend_right=True, cusum=True)
    ax.get_figure().savefig('../svg/mpl_test_cusum_dask.svg')


def test_matplotlib_cusum_identical():
    svg1 = drop_svg_date('../svg/mpl_test_cusum_no_dask.svg')
    svg2 = drop_svg_date('../svg/mpl_test_cusum_dask.svg')
    # Line by line so we can get just the line that is different if it happens
    for line in range(len(svg1)):
        assert svg1[line] == svg2[line]


def test_control_chart_x_bar_only():
    x = load_df('data/swiss_real.csv')
    with pytest.raises(ValueError):
        ax = control_chart(x, x_bar=[(1, 2, 3)], alpha=0.01, legend_right=True, cusum=True)
