# Pre-trade gate. Every order passes through check_order() before submission.
# Fails closed: if a limit can't be verified, the order is blocked.
from alpaca.data.requests import StockLatestTradeRequest

from config.settings import get_settings
from data.connector import get_historical_client, get_trading_client
from data.logger import get_logger

logger = get_logger(__name__)


def _latest_price(symbol: str) -> float:
    client = get_historical_client()
    trade = client.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
    return float(trade.price)


def _position_qty(client, symbol: str) -> float:
    try:
        return float(client.get_open_position(symbol).qty)
    except Exception:
        return 0.0


def _portfolio_exposure(client) -> float:
    return sum(abs(float(p.market_value)) for p in client.get_all_positions())


def check_order(symbol: str, side: str, qty: float):
    """Returns (allowed, reason)."""
    s = get_settings()

    try:
        client = get_trading_client()
        account = client.get_account()
        current_qty = _position_qty(client, symbol)
        price = _latest_price(symbol)
        current_exposure = _portfolio_exposure(client)
    except Exception as e:
        logger.error("Risk check could not verify %s: %s", symbol, e)
        return False, f"risk check failed (blocking): {e}"

    notional = qty * price

    # 1. Max shares per symbol
    if side == "buy" and current_qty + qty > s.max_position_shares:
        return False, (f"would hold {current_qty + qty:g} shares of {symbol}, "
                       f"over max_position_shares={s.max_position_shares:g}")

    # 2. Max notional per order
    if notional > s.max_notional_per_order:
        return False, (f"notional ${notional:,.2f} over "
                       f"max_notional_per_order=${s.max_notional_per_order:,.2f}")

    # 3. Buying power
    if side == "buy" and notional > float(account.buying_power):
        return False, (f"notional ${notional:,.2f} over buying power "
                       f"${float(account.buying_power):,.2f}")

    # 4. Portfolio-wide exposure cap
    projected = current_exposure + (notional if side == "buy" else 0)
    cap = s.max_portfolio_exposure_pct * float(account.equity)
    if projected > cap:
        return False, (f"projected exposure ${projected:,.2f} over "
                       f"{s.max_portfolio_exposure_pct:.0%} of equity (${cap:,.2f})")

    return True, "ok"


def check_stop_loss_take_profit(symbol: str, stop_loss_pct: float = None,
                                take_profit_pct: float = None):
    """Returns (should_exit, reason). Caller decides what to do."""
    s = get_settings()
    sl = s.stop_loss_pct if stop_loss_pct is None else stop_loss_pct
    tp = s.take_profit_pct if take_profit_pct is None else take_profit_pct

    try:
        position = get_trading_client().get_open_position(symbol)
    except Exception:
        return False, "no open position"

    plpc = float(position.unrealized_plpc)

    if plpc <= -abs(sl):
        return True, f"stop-loss: {plpc:.2%} <= -{sl:.2%}"
    if plpc >= abs(tp):
        return True, f"take-profit: {plpc:.2%} >= {tp:.2%}"
    return False, f"within bounds ({plpc:.2%})"
