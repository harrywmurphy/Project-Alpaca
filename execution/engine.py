# Signal -> risk gate -> order -> fill, looped over the ticker universe.
import argparse
import os

from config.settings import get_settings
from data.connector import get_trading_client
from data.historical import fetch_bars
from data.logger import get_logger
from execution import order_manager
from execution.order_manager import OrderStore
from risk.limits import check_order, check_stop_loss_take_profit
from strategy.model import STRATEGIES

logger = get_logger(__name__)


def make_strategy_signal_fn(strategy_name: str, days: int = 365):
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Options: {list(STRATEGIES)}")
    fn = STRATEGIES[strategy_name]

    def get_signal(symbol):
        df = fetch_bars(symbol, days=days)
        if df.empty:
            raise ValueError(f"No bars for {symbol}")
        return int(fn(df).iloc[-1]), df.index[-1]

    return get_signal


def make_ml_signal_fn(days: int = 400):
    # Per-ticker RF artifacts, loaded once and cached.
    from strategy.ml_model import load_artifacts, predict_latest_signal
    bundles = {}

    def get_signal(symbol):
        bundle = bundles.setdefault(symbol, load_artifacts(symbol))
        signal, proba, bar_date = predict_latest_signal(symbol, days=days, bundle=bundle)
        logger.info("%s ML: P(up)=%.3f -> %s", symbol, proba, "Long" if signal else "Flat")
        return signal, bar_date

    return get_signal


def get_current_position(client, symbol: str) -> float:
    try:
        return float(client.get_open_position(symbol).qty)
    except Exception:
        return 0.0


def _order_store() -> OrderStore:
    s = get_settings()
    return OrderStore(os.path.join(os.path.dirname(s.db_path), "orders.db"))


def run_cycle(symbol: str, signal_fn, qty: float, store: OrderStore,
              wait_for_fill: bool = True) -> dict:
    client = get_trading_client()
    current_qty = get_current_position(client, symbol)

    signal_value, bar_date = signal_fn(symbol)
    label = "Long" if signal_value == 1 else "Flat"

    # Stop-loss / take-profit overrides the signal on open positions.
    forced_exit = False
    if current_qty > 0:
        should_exit, why = check_stop_loss_take_profit(symbol)
        if should_exit:
            logger.warning("%s: %s -- forcing exit", symbol, why)
            signal_value, forced_exit = 0, True

    logger.info("%s | bar=%s signal=%s pos=%s", symbol, bar_date.date(), label, current_qty)

    if signal_value == 1 and current_qty == 0:
        side, order_qty = "buy", qty
    elif signal_value == 0 and current_qty > 0:
        side, order_qty = "sell", current_qty
    else:
        return {"symbol": symbol, "date": bar_date, "signal": label,
                "position": current_qty, "action": None}

    allowed, reason = check_order(symbol, side, order_qty)
    if not allowed:
        logger.warning("%s: %s blocked -- %s", symbol, side, reason)
        return {"symbol": symbol, "date": bar_date, "signal": label,
                "position": current_qty, "action": side, "blocked": reason}

    order = order_manager.submit_order(symbol, side, order_qty, store=store)
    if order is None:
        return {"symbol": symbol, "date": bar_date, "signal": label,
                "position": current_qty, "action": side, "order_id": None}

    status = None
    if wait_for_fill:
        final = order_manager.poll_order_status(order.id, store=store)
        status = getattr(getattr(final, "status", None), "value", None)

    return {"symbol": symbol, "date": bar_date, "signal": label,
            "position": current_qty, "action": side, "forced_exit": forced_exit,
            "order_id": str(order.id), "status": status}


def run_all(strategy_name: str = None, qty: float = None, wait_for_fill: bool = True,
            store: OrderStore = None, use_ml: bool = False) -> list:
    s = get_settings()
    strategy_name = strategy_name or s.strategy
    qty = s.default_qty if qty is None else qty
    store = store or _order_store()

    signal_fn = make_ml_signal_fn() if use_ml else make_strategy_signal_fn(strategy_name)

    results = []
    for symbol in s.tickers:
        try:
            results.append(run_cycle(symbol, signal_fn, qty, store, wait_for_fill))
        except Exception as e:
            logger.exception("%s: cycle failed: %s", symbol, e)
            results.append({"symbol": symbol, "error": str(e)})
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One paper-trading cycle across the universe.")
    p.add_argument("--strategy", default=None, choices=list(STRATEGIES))
    p.add_argument("--qty", type=float, default=None)
    p.add_argument("--ml", action="store_true", help="Use trained Random Forest instead of rules.")
    p.add_argument("--no-wait", action="store_true", help="Don't block on fill confirmation.")
    args = p.parse_args()

    for r in run_all(args.strategy, args.qty, not args.no_wait, use_ml=args.ml):
        print(r)
