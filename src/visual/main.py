from typing import List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

def plot_trades(time_series: pd.Series, levels: List, trades: dict, figsize: tuple = (10, 10)) -> Axes:
    """
    Plot spread with upper/lower thresholds and entry/exit markers.

    :param time_series: (pd.Series) Data to plot
    :param levels: (list) Horizontal levels to plot in the chart
    :param trades: (dict) Trading data contianed in a dictionary
    :param figsize: (tuple) Tuple describing the size of the plot.
    :return: (Axes) Axes object.
    """

    fig = plt.figure(figsize = figsize)
    ax_object = fig.add_subplot()

    # Plot time series data
    ax_object.plot(time_series)

    # Plot horizontal levels
    for level in levels:
        ax_object.plot(time_series.index, np.repeat(level, len(time_series)))

    # Plot trades
    for entry, trade in trades.items():
        if trade['side'] > 0:
            ax_object.scatter(
                entry,
                time_series.loc[entry],
                marker="^",
                s=70,
                color="green",
                label="long"
            )
        else:
            ax_object.scatter(
                entry,
                time_series.loc[entry],
                marker="v",
                s=70,
                color="red",
                label="short"
            )

        ax_object.scatter(
                trade['t1'],
                time_series.loc[trade['t1']],
                marker=".",
                s=70,
                color="black",
                label="exit"
            )

    return ax_object


def plot_equity(equity: pd.Series, title: str = 'Equity Curve', figsize: tuple = (12, 6)) -> Axes:
    """
    Plots an equity curve.

    :param equity: (pd.Series) Data to plot
    :param title: (str) Figure title
    :param figsize: (tuple) Tuple describing the size of the plot.
    :return: (Axes) Axes object.
    """
    fig = plt.figure(figsize = figsize)
    ax_object = fig.add_subplot()

    # Plot time series data
    ax_object.plot(equity,  color='firebrick')

    ax_object.set_title(title)
    # ax_object.legend()
    ax_object.grid()

    return ax_object
