"""
This module allows simulation of cointegrated time series pairs.
"""
from typing import Tuple, Optional

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class CointegrationSimulation:
    """
    This is a class that can be used to simulate cointegrated price series pairs.

    The class will generate a price first-order difference time series defined by an AR(1) process,
    a cointegration error series defined by an AR(1) process, and calculate the other price series
    based on the cointegration equation.
    """

    def __init__(self, ts_num: int, ts_length: int):
        """
        Initialize the simulation class.

        Specify the number of time series to be simulated and define the length of each time series.
        Generate a default parameter set with the initialize_params method.

        :param ts_num: (int) Number of time series to simulate.
        :param ts_length: (int) Length of each time series to simulate.
        """

        self.ts_num = ts_num
        self.ts_length = ts_length
        self.__price_params, self.__coint_params = self.initialize_params()


    def load_params(self, params: dict, target: str = "price"):
        """
        Setter for simulation parameters.

        Change the entire parameter sets by loading the dictionary.

        :param params: (dict) Parameter dictionary.
        :param target: (str) Indicate which parameter to load. Possible values are "price" and "coint".
        """

        # Check which parameters to change
        target_types = ('price', 'coint')
        if target not in target_types:
            raise ValueError("Invalid parameter dictionary type. Expect one of: {}".format(target_types))

        # Check if all necessary parameters are in the provided dictionary
        if target == "price":
            default_keys = set(self.__price_params.keys())
        else:
            default_keys = set(self.__coint_params.keys())

        new_keys = set(params.keys())
        if not default_keys <= new_keys:
            missing_keys = default_keys - new_keys
            raise KeyError("Key parameters {} missing.".format(*missing_keys))

        # Set the parameters
        if target == "price":
            self.__price_params = params
        else:
            self.__coint_params = params

    def simulate_coint(self, initial_price: float) -> Tuple[np.array, np.array, np.array]:
        """
        Generate cointegrated price series and cointegration error series.

        :param initial_price: (float) Starting price of share S2.
        :return: (np.array, np.array, np.array) Price series of share S1, price series of share S2,
            and cointegration error.
        """

        # Read the parameters from the param dictionary
        beta = self.__coint_params['beta']

        share_s2_diff = self.simulate_ar(self.__price_params, use_statsmodels = True)

        # Do a cumulative sum to get share s2 price for each column
        share_s2 = initial_price + np.cumsum(share_s2_diff, axis=0)

        # Now generate the cointegration series
        coint_error = self.simulate_ar(self.__coint_params, use_statsmodels = True)

        # Generate share s1 price according to the cointegration relation
        share_s1 = coint_error - beta * share_s2

        return share_s1, share_s2, coint_error


    def simulate_ar(self, params: dict, burn_in: int = 50, use_statsmodels: bool = True) -> np.array:
        """
        Simulate an AR(1) process without using the statsmodels package.
        The AR(1) process is defined as the following recurrence relation.

        .. math::
            y_t = \\mu + \\phi y_{t-1} + e_t, \\quad e_t \\sim N(0, \\sigma^2) \\qquad \\mathrm{i.i.d}

        :param params: (dict) A parameter dictionary containing AR(1) coefficient, constant trend,
            and white noise variance.
        :param burn_in: (int) The amount of data used to burn in the process.
        :param use_statsmodel: (bool) If True, use statsmodels;
            otherwise, directly calculate recurrence.
        :return: (np.array) ts_num simulated series generated.
        """

        # Store the series
        series_list = []

        # Read the parameters from the dictionary
        try:
            constant_trend = params['constant_trend']
            ar_coeff = params['ar_coeff']
            white_noise_var = params['white_noise_var']
        except KeyError as bad_input:
            raise KeyError("Missing crucial parameters. The parameter dictionary should contain"
                           " the following keys:\n"
                           "1. constant_trend\n"
                           "2. ar_coeff\n"
                           "3. white_noise_var\n"
                           "Call initialize_params() to reset the configuration of the "
                           "parameters to default.") from bad_input

        # If using statsmodels
        if use_statsmodels:
            # Specify AR(1) coefficient
            ar = np.array([1, -ar_coeff])

            # No MA component, but need a constant
            ma = np.array([1])

            # Initialize an ArmaProcess model
            process = sm.tsa.ArmaProcess(ar, ma)

            # Generate the samples
            ar_series = process.generate_sample(nsample=(self.ts_length, self.ts_num),
                                                burnin=burn_in,
                                                scale=np.sqrt(white_noise_var))

            # Add constant trend
            ar_series += constant_trend

            return ar_series

        for _ in range(self.ts_num):
            # Setting an initial point. It does not matter due to the burn-in process.
            # We just need to get the recurrence started.
            series = [np.random.normal()]

            # Now set up the recurrence
            for _ in range(self.ts_length + burn_in):
                y_new = constant_trend + ar_coeff * series[-1] + np.random.normal(0, np.sqrt(white_noise_var))
                series.append(y_new)

            # Reshape the 1-D array into a matrix
            final_series = np.array(series[(burn_in + 1):]).reshape(-1, 1)

            # Use hstack to get the full matrix
            series_list.append(final_series)

        if self.ts_num == 1:
            return series_list[0]
        return np.hstack(tuple(series_list))

    @staticmethod
    def initialize_params() -> Tuple[dict, dict]:
        """
        Initialize the default parameters for the first-order difference of share S2 price series
        and cointegration error.

        :return: (dict, dict) Necessary parameters for share S2 price simulation;
            necessary parameters for cointegration error simulation.
        """

        price_params = {
            "ar_coeff": 0.1,
            "white_noise_var": 0.5,
            "constant_trend": 13.}

        coint_params = {
            "ar_coeff": 0.2,
            "white_noise_var": 1.,
            "constant_trend": 13.,
            "beta": -0.2}

        return price_params, coint_params


    def plot(self, series_x: np.array, series_y: np.array, coint_error: np.array,
                          figw: float = 15., figh: float = 10.) -> plt.Figure:
        """
        Plot the simulated cointegrated series.

        :param series_x: (np.array) Price series of share S1
        :param series_y: (np.array) price series of share S2
        :param coint_error: (np.array) Cointegration error.
        :param figw: (float) Figure width.
        :param figh: (float) Figure height.
        :return: (plt.Figure) Figure with the simulated cointegrated series.
        """

        # Creating a plot
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(figw, figh),
                                       gridspec_kw={'height_ratios': [2.5, 1]})

        # Plot prices
        ax1.plot(series_x, label="Share S1")
        ax1.plot(series_y, label="Share S2")
        ax1.legend(loc='best', fontsize=12)
        ax1.tick_params(axis='y', labelsize=14)

        # Plot cointegration error
        ax2.plot(coint_error, label='spread')
        ax2.legend(loc='best', fontsize=12)

        # Plot the title
        fig.suptitle(r"Simulated cointegrated series and the cointegration error, "
                     r"$\beta = {}$".format(self.__coint_params['beta']), fontsize=20)

        return fig
