
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

class PairsTrading:
    """
    The class implements the consturction of a vectorized backtest methodology for
    pairs trading.
    """

    def __init__(self, spread: pd.Series, poistions = pd.Series):
        """
        Class constructor method.
        """
        self.portfolio = spread
        self.positions = poistions

    def plot_stratgy(self, figsize: Tuple = (12, 9)) -> Axes:
        """
        Plots the proflio price, positions and equity curve of the strategy

        :param figsize: (tuple) Tuple describing the size of the plot.
        :return: (Axes) Axes object.
        """
        pnl = self.portfolio.diff()

        pnl_strat = pnl * self.positions

        equity = pnl_strat.cumsum()

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Portfolio
        axes[0].plot(self.portfolio.index, self.portfolio.values, color = 'red')
        axes[0].set_title("Portfolio Price")
        axes[0].grid(True, alpha=0.3)

        # Positions
        axes[1].step(self.positions.index, self.positions.values, where="post", color = 'orange')
        axes[1].set_title("Positions")
        axes[1].grid(True, alpha=0.3)

        # 3 Equity curve
        axes[2].plot(equity.index, equity.values)
        axes[2].set_title("Equity Curve: P&L")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return axes
