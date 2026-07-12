# Rule-based strategies. Each returns a 0/1 long-only exposure series.
import numpy as np
import pandas as pd

from strategy import features as ind


def _statefulize(entry: pd.Series, exit_: pd.Series, index) -> pd.Series:
    # Hold position between entry and exit rather than firing on single bars.
    signal = pd.Series(np.nan, index=index)
    signal[entry] = 1.0
    signal[exit_] = 0.0
    return signal.ffill().fillna(0).astype(int)


def trend_following(df: pd.DataFrame) -> pd.Series:
    # MACD cross, only when ADX confirms a real trend.
    m = ind.macd(df["close"])
    a = ind.adx(df["high"], df["low"], df["close"])
    return _statefulize((m["macd"] > m["signal"]) & (a > 25),
                        m["macd"] < m["signal"], df.index)


def mean_reversion(df: pd.DataFrame) -> pd.Series:
    # Oversold + below lower band -> buy the dip.
    r = ind.rsi(df["close"])
    bb = ind.bollinger_bands(df["close"])
    return _statefulize((r < 30) & (df["close"] < bb["lower"]),
                        (r > 70) & (df["close"] > bb["upper"]), df.index)


def custom_strategy(df: pd.DataFrame) -> pd.Series:
    # Long-term trend filter + momentum + money flow.
    e200 = ind.ema(df["close"], 200)
    r = ind.rsi(df["close"])
    c = ind.cmf(df["high"], df["low"], df["close"], df["volume"])
    return _statefulize((df["close"] > e200) & (r > 40) & (c > 0),
                        (df["close"] < e200) | (r > 70), df.index)


STRATEGIES = {
    "Trend Following": trend_following,
    "Mean Reversion": mean_reversion,
    "Custom": custom_strategy,
}
