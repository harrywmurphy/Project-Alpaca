# Historical OHLCV bars + candlestick plot.
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from data.connector import get_historical_client

# Free feed can't serve the last 15 min.
_DELAY = timedelta(minutes=16)


def fetch_bars(ticker: str, days: int = 365, timeframe=TimeFrame.Day):
    client = get_historical_client()
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=timeframe,
        start=datetime.now() - timedelta(days=days),
        end=datetime.now() - _DELAY,
        feed=DataFeed.IEX,
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return df
    return df.reset_index(level=0, drop=True)  # drop symbol level, keep timestamp


def get_day_data(df, selected_date):
    return df[df.index.date == selected_date]


def build_plot(df):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=len(df) > 30, height=550,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
