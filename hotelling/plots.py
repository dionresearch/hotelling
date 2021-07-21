# -*- coding: utf-8 -*-
"""plot.py.

Hotelling's T-Squared multivariate control charts

See:

  - Hotelling, Harold. (1931). The Generalization of Student's Ratio. Ann. Math. Statist. 2,
    no. 3, 360--378. doi:10.1214/aoms/1177732979.
  - Tukey, J. W. (1960). A survey of sampling from contaminated distributions. In: Contributions
    to Probability and Statistics. Stanford Univ. Press. 448-85
  - Gnanadesikan, R. and J.R. Kettenring (1972). Robust Estimates, Residuals, and Outlier Detection
    with Multiresponse Data. Biometrics 28, 81-124

"""
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd

try:
    from plotly.offline import iplot
    from plotly.subplots import make_subplots
    import plotly.tools as tls

    plotly_module = True
except ModuleNotFoundError:
    plotly_module = False
from scipy import stats

from hotelling.stats import hotelling_t2


def control_interval(m, n, f, phase=1, alpha=0.001):
    """control_interval.

    For Hotelling control charts, phase 1 is using Qi. This follows a beta distribution, not an F distribution. For
    phase 2 uses future observations. These would follow a known distribution ~ F (Seber, 1984).
    The lower and upper lines are based on the quantiles of the distribution (aka `percent point function`)
    for α and 1 - α, while the center line is the median (50%).

    See:
        - Seber, G (1984). Multivariate Observations. John Wiley & Sons.
        - Nola D. Tracy, John C. Young & Robert L. Mason (1992) Multivariate Control Charts for individual Observations,
          Journal or Quality Technology, 24:2, 88-95, DOI:10.1080/00224065.1992.12015232

    :param m: sample groups (between 1 and n)
    :param n: number of samples
    :param f: number of features in the multivariate samples
    :param phase: 1 or 2 - phase 1 is within initial sample, phase 2 is measuring implemented control
    :param alpha: significance level - used to calculate control lines at α/2 and 1-α/2
    :return:
    """
    if phase == 1:
        lcl = float(
            ((m - 1) * (n - 1) / m)
            * (stats.beta(f / 2, ((m - f - 1) / 2)).ppf(alpha / 2)),
        )
        cl = float(
            ((m - 1) * (n - 1) / m) * (stats.beta(f / 2, ((m - f - 1) / 2)).ppf(0.5)),
        )
        ucl = float(
            ((m - 1) * (n - 1) / m)
            * (stats.beta(f / 2, ((m - f - 1) / 2)).ppf(1 - alpha / 2)),
        )
    else:
        lcl = float(
            (f * (m - 1) * (m + 1)) / (m * (m - f)) * stats.f(f, m - f).ppf(alpha / 2)
        )
        cl = float((f * (m - 1) * (m + 1)) / (m * (m - f)) * stats.f(f, m - f).ppf(0.5))
        ucl = float(
            (f * (m - 1) * (m + 1))
            / (m * (m - f))
            * stats.f(f, m - f).ppf(1 - alpha / 2)
        )
    return lcl, cl, ucl


def control_stats(x):
    """control_stats.

    Compute the sample mean vector and the covariance matrix

    :param x: pandas dataframe, uni or multivariate
    :return: sample mean, sample covariance
    """
    try:
        return x.mean(0).compute(), x.cov().compute()
    except AttributeError:
        return x.mean(0), x.cov()


def control_chart(
    x,
    phase=1,
    alpha=0.001,
    x_bar=None,
    s=None,
    legend_right=False,
    interactive=False,
    width=10,
    cusum=False,
    template="none",
    marker="o",
    ooc_marker="x",
    random_state=42,
    limit=1000,
    no_display=False,
):
    """control_chart.

    Hotelling Control Chart based on Q / T^2.

    See also `control_interval` for more detail

    :param x: pandas dataframe, uni or multivariate
    :param phase: 1 or 2 - phase 1 is within initial sample, phase 2 is measuring implemented control
    :param alpha: significance level - used to calculate control lines at α/2 and 1-α/2
    :param x_bar: sample mean (optional, required with s)
    :param s: sample covariance (optional, required with x_bar)
    :param legend_right: default to 'left', can specify 'right'
    :param interactive: if  True and plotly is available, renders as interactive plot in notebook. False, render image.
    :param width: how many units wide. defaults to 10, good for notebooks
    :param cusum: use cumulative sum instead of average
    :param template: plotly template, defaults to 'none', matching default matplotlib
    :param marker: default marker symbol - one valid for matplotlib
    :param ooc_marker: out of control marker symbol (x) - one valid for matplotlib
    :param random_state: seed for sample (n > limit)
    :param limit: max number of points to plot, defaults to 1000
    :return: matplotlib ax / plotly fig
    """
    n, subset = limit_display(x, limit, random_state)
    m = n

    # computing each individual values to the mean and covariance of the whole dataset
    if x_bar is None and s is None:
        x_bar, s = control_stats(x)
    elif x_bar is None or s is None:
        raise ValueError("Error: must specify both x_bar and s, or none at all.")

    # data might be a subset (sample), but control stats above are calculated on the whole dataset
    points, f = subset.shape
    qi = [hotelling_t2(subset[i:i + 1], x_bar, S=s) for i in range(points)]

    df = pd.DataFrame({"qi": qi})

    lcl, cl, ucl = control_interval(m, n, f, phase=phase, alpha=alpha)

    cusum_text = ""
    if cusum:
        df["deviation"] = (df["qi"] - cl).cumsum() + cl
        cusum_text = f" w/deviation (ref={cl:.3f})"
    ax = df.plot(
        title=f"Hotelling Control Chart (α={alpha}, phase={phase}{cusum_text})",
        marker=marker,
        figsize=(width, width / 2),
    )
    ax.set_xlabel("samples")

    try:
        df[(df["qi"] > ucl) | (df["qi"] < lcl)].plot(
            ax=ax, marker=ooc_marker, linestyle="None", color="red", legend=None
        )
    except TypeError:
        # nothing to plot
        pass
    x_pos = 0
    align = "left"
    if legend_right:
        x_pos = len(qi)
        align = "right"
    font_dict = {"family": "serif", "color": "red", "size": 10}
    if not interactive:
        ax.hlines(
            ucl,
            xmin=0,
            xmax=len(qi),
            linestyles="dashed",
            color="r",
            label=f"UCL={ucl}",
        )
    plt.text(
        x_pos,
        ucl + 0.1,
        s=f"UCL={ucl:.3f}",
        fontdict=font_dict,
        horizontalalignment=align,
    )
    if not interactive:
        ax.hlines(
            cl, xmin=0, xmax=len(qi), linestyles="dashed", color="k", label=f"CL={cl}"
        )
    font_dict = {"family": "serif", "color": "black", "size": 10}
    plt.text(
        x_pos, cl + 0.1, s=f"CL={cl:.3f}", fontdict=font_dict, horizontalalignment=align
    )
    if not interactive:
        ax.hlines(
            lcl,
            xmin=0,
            xmax=len(qi),
            linestyles="dashed",
            color="r",
            label=f"LCL={lcl}",
        )
    font_dict = {"family": "serif", "color": "red", "size": 10}
    plt.text(
        x_pos,
        lcl + 0.1,
        s=f"LCL={lcl:.3f}",
        fontdict=font_dict,
        horizontalalignment=align,
    )
    if plotly_module and interactive:
        fig = tls.mpl_to_plotly(ax.get_figure())
        for var, col in [(ucl, "Red"), (lcl, "Red"), (cl, "Black")]:
            fig.add_shape(
                type="line",
                x0=0,
                y0=var,
                x1=len(qi),
                y1=var,
                line=dict(color=col, width=4, dash="dashdot",),
            )
        fig.update_layout(template=template)
        if no_display is False:
            iplot(fig)
        return fig
    else:
        return ax


def limit_display(x, limit, random_state):
    """limit_displau.

    Convenient way to get around the issue of very large datasets. We can't show everything, so we display
    a subset. The tests and stats like T2, F and P values are not affected, because we calculate them on all
    the data.

    :param x: dask or pandas dataframe, uni or multivariate
    :param random_state: seed for sample (n > limit)
    :param limit: max number of points to plot, defaults to 1000
    :return: returns original number of rows and limited dataframe
    """
    n, *f = x.shape

    try:
        n = n.compute()
    except AttributeError:
        pass
    if n > limit:
        try:
            frac = 1000 / n
            subset = x.sample(frac=frac, random_state=random_state).compute()
        except AttributeError:
            subset = x.sample(n=1000, random_state=random_state)
    else:
        # The whole thing
        try:
            subset = x.compute()
        except AttributeError:
            subset = x

    return n, subset


def univariate_control_chart(
    x,
    var=None,
    sigma=3,
    legend_right=False,
    interactive=False,
    connected=True,
    width=10,
    cusum=False,
    cusum_only=False,
    template="none",
    marker="o",
    ooc_marker="x",
    limit=1000,
    random_state=42,
    no_display=False,
):
    """univariate_control_chart.

    :param x: dask or pandas dataframe, uni or multivariate
    :param var: optional, variable to plot (default to all)
    :param sigma: default to 3 sigma from mean for upper and lower control lines
    :param legend_right: default to 'left', can specify 'right'
    :param interactive: if plotly is available, renders as interactive plot in notebook. False to render image.
    :param connected: defaults to True. Appropriate when time related /consecutive batches, else, should be False
    :param width: how many units wide. defaults to 10, good for notebooks
    :param cusum: use cumulative sum instead of average
    :param cusum_only: don't display values, just cusum referenced to 0
    :param template: plotly template, defaults to 'none', matching default matplotlib
    :param marker: default marker symbol (o) - one valid for matplotlib
    :param ooc_marker: out of control marker symbol (x) - one valid for matplotlib
    :param random_state: seed for sample (n > limit)
    :param limit: max number of points to plot, defaults to 1000
    :return: returns matplotlib figure or array of plotly figures
    """
    n, *f, df = limit_display(x, limit, random_state)
    num_plots = len(df.columns)
    k = sigma  # 3 sigma default
    if interactive:
        fig = make_subplots(rows=num_plots, cols=1)
    else:
        fig = plt.figure(figsize=(width, num_plots * width / 2))

    ax = list(range(num_plots))

    layout = num_plots * 100 + 11
    features = df.columns if var is None else [var]
    x_pos = 0
    align = "left"
    if legend_right:
        x_pos = n
        align = "right"

    plotly_figs = []
    for i, var in enumerate(features):
        x_bar = df[var].mean()
        cusum_text = ""
        columns = var
        if cusum:
            if cusum_only:
                columns = "deviation"
                df["deviation"] = (df[var] - x_bar).cumsum()
                cusum_text = f" cumulative deviation (ref={x_bar:.3f})"
            else:
                columns = [var, "deviation"]
                df["deviation"] = (df[var] - x_bar).cumsum() + (x_bar)
                cusum_text = f" w/deviation (ref={x_bar:.3f})"
        ucl = x_bar + k * df[var].std()
        lcl = x_bar - k * df[var].std()
        if interactive:
            mpl_fig, ax[i] = plt.subplots(figsize=(width, width / 2))
        else:
            ax[i] = fig.add_subplot(layout + i)
        if connected:
            df[columns].plot(ax=ax[i], marker=marker)
        else:
            df[columns].plot(ax=ax[i], marker=marker, linestyle="None")
        try:
            if cusum_only is False:
                df[var][(x[var] > ucl) | (x[var] < lcl)].plot(
                    ax=ax[i], marker=ooc_marker, linestyle="None", color="red"
                )
        except TypeError:
            # no outliers
            pass
        x_min = df.index.min()
        x_max = df.index.max()
        if cusum_only is False:
            y_low = min(df[var].min(), lcl) - 0.1 * abs(df[var].min())
            y_high = max(df[var].max(), ucl) + 0.1 * abs(df[var].max())
        elif cusum:
            y_low = min(df["deviation"].min() - 0.1 * abs(df["deviation"].min()), 0)
            y_high = df["deviation"].max() + 0.1 * abs(df["deviation"].max())
        else:
            warn("Error: must specify cusum=True when using cusum_only=True.")

        if plotly_module and interactive and cusum_only is False:
            ucl_text = dict(
                x=x_pos,
                y=ucl + 0.2,
                showarrow=False,
                text=f"UCL={ucl:.3f}",
                xref="x",
                yref="y",
                font=dict(family="serif", color="red", size=10),
            )
            mean_text = dict(
                x=x_pos,
                y=x_bar + 0.2,
                showarrow=False,
                text=f"mean={x_bar:.3f}",
                xref="x",
                yref="y",
                font=dict(family="serif", color="black", size=10),
            )
            lcl_text = dict(
                x=x_pos,
                y=lcl + 0.2,
                showarrow=False,
                text=f"LCL={lcl:.3f}",
                xref="x",
                yref="y",
                font=dict(family="serif", color="red", size=10),
            )
        elif cusum_only is False:
            ax[i].hlines(
                ucl, xmin=x_min, xmax=x_max, linestyles="dashed", color="r", label="UCL"
            )
            font_dict = {"family": "serif", "color": "red", "size": 10}
            plt.text(
                x_pos,
                ucl + 0.2,
                s=f"UCL={ucl:.3f}",
                fontdict=font_dict,
                horizontalalignment=align,
            )
            ax[i].hlines(
                x_bar,
                xmin=x_min,
                xmax=x_max,
                linestyles="dashed",
                color="k",
                label="mean",
            )
            font_dict = {"family": "serif", "color": "black", "size": 10}
            plt.text(
                x_pos,
                x_bar + 0.2,
                s=f"mean={x_bar:.3f}",
                fontdict=font_dict,
                horizontalalignment=align,
            )

            ax[i].hlines(
                lcl, xmin=x_min, xmax=x_max, linestyles="dashed", color="r", label="LCL"
            )
            font_dict = {"family": "serif", "color": "red", "size": 10}
            plt.text(
                x_pos,
                lcl + 0.2,
                s=f"LCL={ucl:.3f}",
                fontdict=font_dict,
                horizontalalignment=align,
            )

        ax[i].title.set_text(
            f"Univariate Control Chart for {var}{cusum_text} (σ={sigma})"
        )
        plt.tight_layout()
        if plotly_module and interactive:
            pfig = tls.mpl_to_plotly(mpl_fig)
            if cusum_only is False:
                for var, col in [(ucl, "Red"), (lcl, "Red"), (x_bar, "Black")]:
                    pfig.add_shape(
                        type="line",
                        x0=x_min,
                        y0=var,
                        x1=x_max,
                        y1=var,
                        line=dict(color=col, width=4, dash="dashdot",),
                    )
            pfig.update_xaxes(range=(x_min - 1, x_max + 1))
            pfig.update_yaxes(range=(y_low, y_high))
            annotations = None if cusum_only else [ucl_text, mean_text, lcl_text]
            pfig.update_layout(margin=dict(l=1, r=1),  # noqa
                yaxis_tickmode="auto",
                annotations=annotations,
                template=template,
            )
            if no_display is False:
                iplot(pfig)
            plotly_figs.append(pfig)
    if interactive:
        return plotly_figs
    else:
        return fig
