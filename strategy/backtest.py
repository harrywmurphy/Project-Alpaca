# Long-only vectorized backtest.
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    initial_capital: float


def run_backtest(price: pd.Series, signal: pd.Series,
                 initial_capital: float = 100_000, fee_bps: float = 0.0) -> BacktestResult:
    price = price.astype(float)
    signal = signal.reindex(price.index).fillna(0).clip(0, 1)

    position = (signal > 0).astype(int)
    position_lagged = position.shift(1).fillna(0)  # no lookahead

    strat_ret = position_lagged * price.pct_change().fillna(0)

    if fee_bps > 0:
        turnover = position_lagged.diff().abs().fillna(0)
        strat_ret = strat_ret - turnover * (fee_bps / 10_000)

    equity = initial_capital * (1 + strat_ret).cumprod()

    changes = position_lagged.diff().fillna(position_lagged)
    dates = changes[changes != 0].index
    trades = pd.DataFrame({
        "date": dates,
        "action": np.where(changes.loc[dates] > 0, "BUY", "SELL"),
        "price": price.loc[dates].values,
    }).reset_index(drop=True)

    return BacktestResult(equity, strat_ret, position_lagged, trades, initial_capital)


def buy_and_hold(price: pd.Series, initial_capital: float = 100_000) -> BacktestResult:
    return run_backtest(price, pd.Series(1, index=price.index), initial_capital)
