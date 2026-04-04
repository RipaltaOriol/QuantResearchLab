from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_OLS_hedge_ratio(price_data: pd.DataFrame, dependent_var = str, add_cosntant: bool = False) -> \
        Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get OLS hedge ratio: y = beta * X.

    :param price_data: (pd.DataFrame) Data Frame with security prices.
    :param dependent_var: (str) Column name which represents the dependent variable (y).
    :param add_constant: (bool) Boolean flag to add constant in regression setting.
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """
    ols = LinearRegression(fit_intercept = add_cosntant)

    X = price_data.copy()
    X.drop(columns = dependent_var, axis=1, inplace=True)
    exogenous_variables = X.columns.tolist()

    if X.shape[1] == 1:
        X = X.values.reshape(-1, 1)

    y = price_data[dependent_var].copy()

    ols.fit(X, y)
    residuals = y - ols.predict(X)

    hedge_ratios = ols.coef_
    hedge_ratios_dict = dict(zip([ols] + exogenous_variables, np.insert(hedge_ratios, 0, 1.0)))

    return hedge_ratios_dict, X, y, residuals
