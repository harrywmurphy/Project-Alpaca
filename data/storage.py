# SQLite tick store. Every streamed quote/trade lands here.
import sqlite3
from contextlib import contextmanager

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, event_type TEXT,
    bid_price REAL, ask_price REAL, bid_size REAL, ask_size REAL,
    trade_price REAL, trade_size REAL,
    timestamp TEXT, collected_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ticks_ticker ON ticks(ticker, collected_at);
"""


class QuoteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def insert_tick(self, row: dict):
        with self._connect() as conn:
            conn.execute("""INSERT INTO ticks
                (ticker, event_type, bid_price, ask_price, bid_size, ask_size,
                 trade_price, trade_size, timestamp, collected_at)
                VALUES (:ticker, :event_type, :bid_price, :ask_price, :bid_size, :ask_size,
                        :trade_price, :trade_size, :timestamp, :collected_at)""", row)

    def get_recent(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM ticks ORDER BY id DESC LIMIT ?", conn, params=(limit,))

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
