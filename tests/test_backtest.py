import numpy as np
import pandas as pd

from strategy.backtest import buy_and_hold, run_backtest


def price():
    return pd.Series([100, 102, 101, 105, 103],
                     index=pd.date_range("2024-01-01", periods=5, freq="D"))


def test_flat_signal_preserves_capital():
    p = price()
    r = run_backtest(p, pd.Series(0, index=p.index), 100_000)
    assert np.isclose(r.equity.iloc[-1], 100_000)


def test_always_long_matches_buy_and_hold():
    p = price()
    a = run_backtest(p, pd.Series(1, index=p.index), 100_000)
    b = buy_and_hold(p, 100_000)
    assert np.isclose(a.equity.iloc[-1], b.equity.iloc[-1])


def test_no_lookahead():
    # Signal turning on at t must not earn t's return.
    p = price()
    sig = pd.Series([0, 1, 1, 1, 1], index=p.index)
    r = run_backtest(p, sig, 100_000)
    assert np.isclose(r.returns.iloc[1], 0.0)


def test_equity_length_matches_price():
    p = price()
    r = run_backtest(p, pd.Series(1, index=p.index), 100_000)
    assert len(r.equity) == len(p)
