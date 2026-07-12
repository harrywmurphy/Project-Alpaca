# Performance stats + plots.
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity): return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity):
    years = len(equity) / TRADING_DAYS
    return np.nan if years <= 0 else (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def volatility(returns): return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns, rf: float = 0.0):
    excess = returns - rf / TRADING_DAYS
    return np.nan if excess.std() == 0 else np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()


def sortino(returns, rf: float = 0.0):
    # Penalizes downside deviation only.
    excess = returns - rf / TRADING_DAYS
    dd = excess[excess < 0].std()
    return np.nan if dd == 0 or np.isnan(dd) else np.sqrt(TRADING_DAYS) * excess.mean() / dd


def drawdown_series(equity): return equity / equity.cummax() - 1


def max_drawdown(equity): return drawdown_series(equity).min()


def win_rate(returns):
    active = returns[returns != 0]  # flat days excluded
    return np.nan if active.empty else (active > 0).mean()


def summarize(result) -> dict:
    eq, ret = result.equity, result.returns
    return {
        "Total Return": total_return(eq),
        "CAGR": cagr(eq),
        "Volatility": volatility(ret),
        "Sharpe": sharpe(ret),
        "Sortino": sortino(ret),
        "Max Drawdown": max_drawdown(eq),
        "Win Rate": win_rate(ret),
        "Trades": len(result.trades),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({name: summarize(r) for name, r in results.items()}).T


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate"]:
        out[c] = df[c].map(lambda x: f"{x:.2%}")
    for c in ["Sharpe", "Sortino"]:
        out[c] = df[c].map(lambda x: f"{x:.2f}")
    out["Trades"] = df["Trades"].astype(int)
    return out


def plot_equity_curve(results: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=label)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(result, label: str = "Strategy"):
    dd = drawdown_series(result.equity)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax.set_title(f"Drawdown: {label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
