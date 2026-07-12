import numpy as np
import pandas as pd
import pytest

from strategy import features as ind


@pytest.fixture
def ohlcv():
    np.random.seed(0)
    n = 300
    close = pd.Series(100 + np.random.randn(n).cumsum(),
                      index=pd.date_range("2024-01-01", periods=n, freq="D"))
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": pd.Series(np.random.randint(1e5, 1e6, n), index=close.index),
    })


def test_rsi_bounded(ohlcv):
    r = ind.rsi(ohlcv["close"]).dropna()
    assert r.between(0, 100).all()


def test_bollinger_ordering(ohlcv):
    bb = ind.bollinger_bands(ohlcv["close"]).dropna()
    assert (bb["upper"] >= bb["mid"]).all() and (bb["mid"] >= bb["lower"]).all()


def test_macd_hist_is_difference(ohlcv):
    m = ind.macd(ohlcv["close"])
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_adx_non_negative(ohlcv):
    a = ind.adx(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
    assert (a >= 0).all()
