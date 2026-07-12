# Order submission + lifecycle. Every order state lands in SQLite.
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from data.connector import get_trading_client
from data.logger import get_logger

logger = get_logger(__name__)

TERMINAL_STATES = {OrderStatus.FILLED, OrderStatus.CANCELED,
                   OrderStatus.EXPIRED, OrderStatus.REJECTED}
NETWORK_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class OrderStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY, ticker TEXT, side TEXT, qty REAL,
                status TEXT, filled_qty REAL, filled_avg_price REAL,
                updated_at TEXT, error TEXT)""")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, order=None, *, order_id=None, ticker=None, side=None,
             qty=None, status=None, error=None):
        row = {
            "order_id": str(order.id) if order else order_id,
            "ticker": order.symbol if order else ticker,
            "side": order.side.value if order else side,
            "qty": float(order.qty) if order and order.qty else qty,
            "status": order.status.value if order else status,
            "filled_qty": float(order.filled_qty) if order and order.filled_qty else None,
            "filled_avg_price": float(order.filled_avg_price) if order and order.filled_avg_price else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        with self._connect() as conn:
            conn.execute("""INSERT INTO orders VALUES
                (:order_id, :ticker, :side, :qty, :status, :filled_qty,
                 :filled_avg_price, :updated_at, :error)
                ON CONFLICT(order_id) DO UPDATE SET
                    status=:status, filled_qty=:filled_qty,
                    filled_avg_price=:filled_avg_price, updated_at=:updated_at,
                    error=:error""", row)

    def get_recent(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM orders ORDER BY updated_at DESC LIMIT ?", conn, params=(limit,))


def submit_order(ticker, side, qty, order_type="market", limit_price=None,
                 time_in_force="day", store=None, max_retries=3):
    # Retries on network errors only. API rejections are terminal.
    client = get_trading_client()
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    tif = TimeInForce[time_in_force.upper()]

    if order_type == "limit":
        req = LimitOrderRequest(symbol=ticker, qty=qty, side=side_enum,
                                time_in_force=tif, limit_price=limit_price)
    else:
        req = MarketOrderRequest(symbol=ticker, qty=qty, side=side_enum, time_in_force=tif)

    for attempt in range(1, max_retries + 2):
        try:
            order = client.submit_order(req)
            logger.info("Order submitted: %s %s x%s -> %s", side, ticker, qty, order.status.value)
            if store:
                store.save(order)
            return order

        except APIError as e:
            logger.error("Order rejected: %s %s x%s (%s)", side, ticker, qty, e)
            if store:
                store.save(order_id=f"rejected-{time.time()}", ticker=ticker, side=side,
                           qty=qty, status="rejected", error=str(e))
            return None

        except NETWORK_ERRORS as e:
            logger.warning("Network error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt > max_retries:
                return None
            time.sleep(2 * attempt)  # backoff


def poll_order_status(order_id, store=None, poll_interval=2, timeout=120):
    # Poll until terminal state or timeout.
    client = get_trading_client()
    deadline = time.monotonic() + timeout
    last_order, net_errors = None, 0

    while True:
        try:
            order = client.get_order_by_id(order_id)
            last_order, net_errors = order, 0
            logger.info("Order %s status=%s filled=%s", order_id, order.status.value, order.filled_qty)
            if store:
                store.save(order)
            if order.status in TERMINAL_STATES:
                return order

        except NETWORK_ERRORS as e:
            net_errors += 1
            logger.warning("Network error polling %s (%d): %s", order_id, net_errors, e)
            if net_errors >= 5:
                return last_order

        if time.monotonic() >= deadline:
            logger.warning("Timed out on order %s", order_id)
            return last_order

        time.sleep(poll_interval)


def cancel_order(order_id, store=None):
    client = get_trading_client()
    try:
        client.cancel_order_by_id(order_id)
        order = client.get_order_by_id(order_id)
        if store:
            store.save(order)
        return order
    except APIError as e:
        logger.error("Could not cancel %s: %s", order_id, e)
        return None
