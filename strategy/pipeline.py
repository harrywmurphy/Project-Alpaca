# Ties data -> signal -> backtest for one ticker.
import pandas as pd

from data.historical import fetch_bars
from strategy.backtest import BacktestResult, buy_and_hold, run_backtest
from strategy.metrics import comparison_table
from strategy.model import STRATEGIES


def run_strategy_backtest(ticker: str, strategy_name: str, days: int = 365,
                          initial_capital: float = 100_000, fee_bps: float = 0.0) -> BacktestResult:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Options: {list(STRATEGIES)}")
    df = fetch_bars(ticker, days=days)
    if df.empty:
        raise ValueError(f"No bars for {ticker}")
    signal = STRATEGIES[strategy_name](df)
    return run_backtest(df["close"], signal, initial_capital, fee_bps)


def run_all_strategies(ticker: str, days: int = 365, initial_capital: float = 100_000,
                       fee_bps: float = 0.0, include_buy_and_hold: bool = True) -> dict:
    # name -> BacktestResult
    df = fetch_bars(ticker, days=days)
    if df.empty:
        raise ValueError(f"No bars for {ticker}")

    results = {name: run_backtest(df["close"], fn(df), initial_capital, fee_bps)
               for name, fn in STRATEGIES.items()}
    if include_buy_and_hold:
        results["Buy & Hold"] = buy_and_hold(df["close"], initial_capital)
    return results


def compare_strategies(ticker: str, days: int = 365, initial_capital: float = 100_000,
                       fee_bps: float = 0.0) -> pd.DataFrame:
    return comparison_table(run_all_strategies(ticker, days, initial_capital, fee_bps))
