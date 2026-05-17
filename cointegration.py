"""
cointegration.py  —  Engle-Granger cointegration test + hedge ratio estimation.

The Engle-Granger two-step method:
  Step 1:  OLS regression  →  ticker1 = α + β·ticker2 + ε
           β is the hedge ratio (how many units of ticker2 hedge one unit of ticker1).
  Step 2:  ADF test on residuals ε  →  if ε is stationary, the pair is cointegrated.

A stationary residual means the spread will always revert to its mean —
the core assumption statistical arbitrage exploits.
"""

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def run_cointegration_test(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
) -> dict:
    """
    Run Engle-Granger cointegration test on the price pair.

    Parameters
    ----------
    prices  : pd.DataFrame  with columns [ticker1, ticker2]
    ticker1 : str
    ticker2 : str

    Returns
    -------
    dict
        score            : EG test statistic
        pvalue           : p-value of the ADF test on residuals
        critical_values  : [1%, 5%, 10%] critical values
        hedge_ratio      : β from OLS (units of ticker2 per unit of ticker1)
        intercept        : α from OLS
        cointegrated     : True if pvalue < 0.05
    """
    y = prices[ticker1].values
    x = prices[ticker2].values

    # Engle-Granger test
    score, pvalue, critical_values = coint(y, x)

    # OLS to get hedge ratio β and intercept α
    x_const    = add_constant(x)
    ols_result = OLS(y, x_const).fit()
    intercept   = ols_result.params[0]
    hedge_ratio = ols_result.params[1]

    return {
        "score"           : score,
        "pvalue"          : pvalue,
        "critical_values" : critical_values,
        "hedge_ratio"     : hedge_ratio,
        "intercept"       : intercept,
        "cointegrated"    : bool(pvalue < 0.05),
    }