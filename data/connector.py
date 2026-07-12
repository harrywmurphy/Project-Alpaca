# Alpaca clients. Keys from .env, never hard-coded.
import os

from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient


def _keys():
    load_dotenv()
    return os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")


def get_historical_client():
    key, secret = _keys()
    return StockHistoricalDataClient(key, secret)


def get_stream_client():
    key, secret = _keys()
    return StockDataStream(key, secret, feed=DataFeed.IEX)


def get_trading_client():
    key, secret = _keys()
    paper = os.getenv("ALPACA_PAPER", "True").strip().lower() in ("1", "true", "yes")
    return TradingClient(key, secret, paper=paper)


def check_connection() -> dict:
    # Used by the UI status bar.
    try:
        account = get_trading_client().get_account()
        return {
            "connected": True,
            "account_status": str(account.status),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
