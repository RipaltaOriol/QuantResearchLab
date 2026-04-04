import warnings
from scipy.integrate import quad
from scipy.optimize import root_scalar
import scipy.optimize as so
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class OrnsteinUhlenbeck:
    """
    This class implements the algorithm for solving the optimal stopping problem in
    markets with mean-reverting tendencies based on the Ornstein-Uhlenbeck model
    mentioned in the following publication:'Tim Leung and Xin Li Optimal Mean
    reversion Trading: Mathematical Analysis and Practical Applications(November 26, 2015)'
    <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_

    Constructing a portfolio with mean-reverting properties is usually attempted by
    simultaneously taking a position in two highly correlated or co-moving assets and is
    labeled as "pairs trading". One of the most important problems faced by investors is
    to determine when to open and close a position.

    To find the liquidation and entry price levels we formulate an optimal double-stopping
    problem that gives the optimal entry and exit level rules. Also, a stop-loss
    constraint is incorporated into this trading problem and solutions are also provided
    by this module.
    """

    def __init__(self):
        self.theta = None  # Long-term mean
        self.mu = None  # Speed at which the values will regroup around the long-term mean
        self.sigma_square = None  # The amplitude of randomness in the system
        self.delta_t = None  # Delta between observations, calculated in years
        self.c = None  # Transaction costs for liquidating or entering the position
        self.r = None  # Discount rate at the moment of liquidating or entering the position
        self.L = None  # Stop-loss level
        self.B_value = None  # Optimal ratio between two assets
        self.entry_level = None  # Optimal entry levels without and with the stop-loss in respective order
        self.liquidation_level = None  # Optimal exit levels without and with the stop-loss in respective order
        self.data = None  # Training data provided by the user
        self.training_period = None  # Current training period
        self.mll = None  # Maximum log likelihood


    def fit(self, data, data_frequency, discount_rate, transaction_cost, start=None, end=None,
            stop_loss=None):
        """
        Fits the Ornstein-Uhlenbeck model to given data and assigns the discount rates,
        transaction costs and stop-loss level for further exit or entry-level calculation.

        :param data: (np.array/pd.DataFrame) An array with time series of portfolio prices / An array with
            time series of of two assets prices. The dimensions should be either nx1 or nx2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        :param discount_rate: (float/tuple) A discount rate either for both entry and exit time
            or a list/tuple of discount rates with exit rate and entry rate in respective order.
        :param transaction_cost: (float/tuple) A transaction cost either for both entry and exit time
            or a list/tuple of transaction costs with exit cost and entry cost in respective order.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :param stop_loss: (float/int) A stop-loss level - the position is assumed to be closed
            immediately upon reaching this pre-defined price level.
        """

        # Creating variables for the discount rate and transaction cost
        self.r = [0, 0]
        self.c = [0, 0]

        # Setting delta parameter using data frequency
        self._fit_delta(data_frequency=data_frequency)

        # Setting discount rate
        self.r = self._fit_rate_cost(input_data=discount_rate)

        # Setting transaction cost
        self.c = self._fit_rate_cost(input_data=transaction_cost)

        # Setting stop loss level if it's given as a correct data type
        if stop_loss is None or isinstance(stop_loss, (float, int)):
            self.L = stop_loss
        else:
            raise Exception("Wrong stop-loss level data type. Please use float.")

        # Allocating portfolio parameters
        if len(data.shape) == 1:  # If the input is series of prices of a portfolio
            self.fit_to_portfolio(data=data, start=start, end=end)
        elif data.shape[1] == 2:  # If the input is series of prices of assets
            self.fit_to_assets(data=data, start=start, end=end)
        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

    def _fit_data(self, start, end):
        """
        Allocates all the training data parameters
        and fits the OU model to this data.

        Helper function used in self.fit().

        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :return: (np.array) An array of prices to construct a portfolio from.
        """

        # Checking the training data type
        if isinstance(self.data, np.ndarray):
            self.data = self.data.transpose()
            data = self.data
        else:
            # Checking if the starting and ending timestamps were specified
            if all(timestamp is not None for timestamp in [start, end]):

                # Selecting a slice of the dataframe chosen by the user
                data = self.data.loc[start:end]
                # Setting the training interval
                self.training_period = [start, end]
            else:
                data = self.data
                self.training_period = [self.data.first_valid_index(),
                                        self.data.last_valid_index()]

            # Transforming pd.Dataframe into a numpy array
            data = data.to_numpy().transpose()

        return data

    @staticmethod
    def _fit_rate_cost(input_data):
        """
        Sets the value for cost and rate parameters.

        Helper function used in self.fit().

        :param input_data: (float/tuple) Input for cost or rate.
        :return: A tuple of two elements with allocated data for cost/rate
        """

        # If given a single value, it's duplicated for exit and entry levels
        if isinstance(input_data, float):
            parameters = [input_data, input_data]

        # If given two values, they are treated as data for exit and entry levels
        elif isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            parameters = input_data
        else:
            raise Exception("Wrong discount rate or transaction cost data type. "
                            "Please use float or tuple with 2 elements.")

        return parameters

    def _fit_delta(self, data_frequency):
        """
        Sets the value of the delta-t parameter,
        depending on data frequency input.

        Helper function used in self.fit().

        :param data_frequency: (str) Data frequency
            ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        if data_frequency == "D":
            self.delta_t = 1 / 252
        elif data_frequency == "M":
            self.delta_t = 1 / 12
        elif data_frequency == "Y":
            self.delta_t = 1
        else:
            raise Exception("Incorrect data frequency. "
                            "Please use one of the options [\"D\", \"M\", \"Y\"].")

    def fit_to_portfolio(self, data=None, start=None, end=None):
        """
        Fits the Ornstein-Uhlenbeck model to time series
        for portfolio prices.

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """
        # Nullifying the training interval data
        self.training_period = [None, None]

        # Setting the new training data pool
        if data is not None:
            self.data = data

        # Getting the needed data
        portfolio = self._fit_data(start, end)

        # Nullifying optimal entry and exit values during model retraining
        self.entry_level = [None, None]
        self.liquidation_level = [None, None]

        # Fitting the model
        parameters = self.optimal_coefficients(portfolio)

        # Setting the OU model parameters
        self.theta = parameters[0]
        self.mu = parameters[1]
        self.sigma_square = parameters[2]
        self.mll = parameters[3]

    @staticmethod
    def portfolio_from_prices(prices, b_variable):
        """
        Constructs a portfolio based on two given asset prices
        and the relative amount of investment for one of them.

        :param prices: (np.array) An array of prices of the two assets
            used to create a portfolio.
        :param b_variable: (float) A coefficient representing the investment.
            into the second asset, investing into the first one equals one.
        :return: (np.array) Portfolio prices. (p. 11)
        """

        # Calculated as: alpha * Asset_1 - beta * Asset_2
        portfolio_price = ((1 / prices[0][0]) * prices[0][:]
                           - (b_variable / prices[1][0]) * prices[1][:])

        return portfolio_price

    def half_life(self):
        """
        Returns the half-life of the fitted OU process. Half-life stands for the average time that
        it takes for the process to revert to its long term mean on a half of its initial deviation.

        :return: (float) Half-life of the fitted OU process
        """
        # Calculating the half-life
        output = np.log(2) / self.mu

        return output

    def fit_to_assets(self, data=None, start=None, end=None):
        """
        Creates the optimal portfolio in terms of Ornstein-Uhlenbeck model
        from two given time series for asset prices and fits the values
        of the model's parameters. (p.13)

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """
        # Nullifying the training interval data
        self.training_period = [None, None]

        # Setting the new training data pool
        if data is not None:
            self.data = data

        # Getting the needed data
        prices = self._fit_data(start, end)

        # Nullifying optimal entry and exit values during model retraining
        self.entry_level = [None, None]
        self.liquidation_level = [None, None]

        # Lambda function that calculates the optimal OU model coefficients
        # for the portfolio constructed from given prices and any given
        # coefficient B_value
        compute_coefficients = lambda x: self.optimal_coefficients(self.portfolio_from_prices(prices, x))

        # Speeding up the calculations
        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.001, 1, 100)
        res = vectorized(linspace)

        # Picking the argmax of beta
        index = res[3].argmax()

        # Setting the OU model parameters
        self.theta = res[0][index]
        self.mu = res[1][index]
        self.sigma_square = res[2][index]
        self.B_value = linspace[index]
        self.mll = res[3][index]

    def optimal_coefficients(self, portfolio):
        """
        Finds the optimal Ornstein-Uhlenbeck model coefficients depending
        on the portfolio prices time series given.(p.13)

        :param portfolio: (np.array) Portfolio prices.
        :return: (tuple) Optimal parameters (theta, mu, sigma_square, max_LL)
        """

        # Setting bounds
        # Theta  R, mu > 0, sigma_squared > 0
        bounds = ((None, None), (1e-5, None), (1e-5, None))

        theta_init = np.mean(portfolio)

        # Initial guesses for theta, mu, sigma
        initial_guess = np.array((theta_init, 100, 100))

        result = so.minimize(self._compute_log_likelihood, initial_guess,
                             args=(portfolio, self.delta_t), bounds=bounds)

        # Unpacking optimal values
        theta, mu, sigma_square = result.x

        # Undo negation
        max_log_likelihood = -result.fun

        return theta, mu, sigma_square, max_log_likelihood

    @staticmethod
    def _compute_log_likelihood(params, *args):
        """
        Computes the average Log Likelihood. (p.13)

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma_squared.
        :param args: (tuple) All other values that to be passed to self._compute_log_likelihood()
        :return: (float) The average log likelihood from given parameters.
        """

        # Setting given parameters
        theta, mu, sigma_squared = params
        X, dt = args
        n = len(X)

        # Calculating log likelihood
        sigma_tilde_squared = sigma_squared * (1 - np.exp(-2 * mu * dt)) / (2 * mu)

        summation_term = sum((X[1:] - X[:-1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt))) ** 2)

        summation_term = -summation_term / (2 * n * sigma_tilde_squared)

        log_likelihood = (-np.log(2 * np.pi) / 2) \
                         + (-np.log(np.sqrt(sigma_tilde_squared))) \
                         + summation_term

        return -log_likelihood
