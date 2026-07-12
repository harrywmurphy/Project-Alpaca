# Live quote/trade stream on a daemon thread. Writes ticks to SQLite.
import asyncio
import threading
from datetime import datetime, timezone

from data.connector import get_stream_client
from data.logger import get_logger
from data.storage import QuoteStore

logger = get_logger(__name__)

_lock = threading.Lock()
_latest = {}
_active_stream = None
_active_thread = None
_store = None


def _get_store(db_path: str) -> QuoteStore:
    global _store
    if _store is None or _store.db_path != db_path:
        _store = QuoteStore(db_path)
        _store.init_db()
    return _store


def stop_stream():
    global _active_stream, _active_thread
    if _active_stream is not None:
        try:
            _active_stream.stop()  # safe cross-thread
        except Exception as e:
            logger.warning("Error stopping stream: %s", e)
    if _active_thread is not None:
        _active_thread.join(timeout=2)
    _active_stream = None
    _active_thread = None
    logger.info("Stream stopped")


def start_stream(symbols, db_path: str):
    global _active_stream, _active_thread

    stop_stream()  # kill previous connection first (free tier = 1 socket)
    store = _get_store(db_path)

    with _lock:
        _latest.clear()
        for s in symbols:
            _latest[s] = {"bid": None, "ask": None, "last": None, "updated_at": None}

    async def quote_handler(q):
        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            _latest.setdefault(q.symbol, {})
            _latest[q.symbol].update(bid=q.bid_price, ask=q.ask_price, updated_at=now)
        store.insert_tick({
            "ticker": q.symbol, "event_type": "quote",
            "bid_price": q.bid_price, "ask_price": q.ask_price,
            "bid_size": q.bid_size, "ask_size": q.ask_size,
            "trade_price": None, "trade_size": None,
            "timestamp": q.timestamp.isoformat() if q.timestamp else None,
            "collected_at": now,
        })

    async def trade_handler(t):
        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            _latest.setdefault(t.symbol, {})
            _latest[t.symbol].update(last=t.price, updated_at=now)
        store.insert_tick({
            "ticker": t.symbol, "event_type": "trade",
            "bid_price": None, "ask_price": None, "bid_size": None, "ask_size": None,
            "trade_price": t.price, "trade_size": t.size,
            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
            "collected_at": now,
        })

    stream = get_stream_client()

    def runner():
        asyncio.set_event_loop(asyncio.new_event_loop())  # bg thread needs its own loop
        stream.subscribe_quotes(quote_handler, *symbols)
        stream.subscribe_trades(trade_handler, *symbols)
        logger.info("Connecting stream for %s", symbols)
        try:
            stream.run()  # blocks
        except Exception as e:
            logger.exception("Stream crashed: %s", e)

    thread = threading.Thread(target=runner, daemon=True)
    _active_stream = stream
    _active_thread = thread
    thread.start()
    return thread


def get_latest() -> dict:
    with _lock:
        return {sym: dict(v) for sym, v in _latest.items()}


def is_running() -> bool:
    return _active_thread is not None and _active_thread.is_alive()
