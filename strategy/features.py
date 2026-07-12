# Technical indicators. Series in, Series out.
import numpy as np
import pandas as pd


# --- TREND ---
def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "signal": signal_line,
                         "hist": macd_line - signal_line})


def _true_range(high, low, close) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def adx(high, low, close, window: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = _true_range(high, low, close).ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


# --- MOMENTUM ---
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)  # 50 = neutral when undefined


# --- VOLATILITY ---
def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * std, "lower": mid - n_std * std})


def atr(high, low, close, window: int = 14) -> pd.Series:
    return _true_range(high, low, close).ewm(alpha=1 / window, adjust=False).mean()


# --- VOLUME ---
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def cmf(high, low, close, volume, window: int = 20) -> pd.Series:
    rng = (high - low).replace(0, np.nan)  # doji guard
    mfv = (((close - low) - (high - close)) / rng) * volume
    return mfv.rolling(window).sum() / volume.rolling(window).sum()
