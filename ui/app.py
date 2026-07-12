# Command & control dashboard. Run: streamlit run ui/app.py
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import get_settings
from data import streamer
from data.connector import check_connection, get_trading_client
from data.historical import build_plot, fetch_bars, get_day_data
from data.storage import QuoteStore
from execution.engine import run_all
from execution.order_manager import OrderStore
from strategy.metrics import comparison_table, format_table
from strategy.model import STRATEGIES
from strategy.pipeline import run_all_strategies

st.set_page_config(page_title="Project Alpaca", layout="wide")

S = get_settings()
ORDERS_DB = os.path.join(os.path.dirname(S.db_path), "orders.db")


# --- status bar ---
conn = check_connection()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Broker", "Connected" if conn["connected"] else "Disconnected")
c2.metric("Mode", "PAPER")
c3.metric("Equity", f"${conn['equity']:,.0f}" if conn["connected"] else "—")
c4.metric("Buying Power", f"${conn['buying_power']:,.0f}" if conn["connected"] else "—")
if not conn["connected"]:
    st.error(conn.get("error", "No broker connection. Check .env keys."))
    st.stop()

tab_ctl, tab_live, tab_bt, tab_cfg = st.tabs(
    ["Control", "Live Data", "Backtest", "Config"])


# --- CONTROL: positions, P&L, orders, run the strategy ---
with tab_ctl:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Run Strategy")
        strat = st.selectbox("Strategy", list(STRATEGIES),
                             index=list(STRATEGIES).index(S.strategy))
        use_ml = st.checkbox("Use ML signal (Random Forest)", value=False,
                             help="Requires artifacts. Run: python train_models.py")
        qty = st.number_input("Order qty", 1, int(S.max_position_shares), int(S.default_qty))
        wait = st.checkbox("Wait for fill", value=True)

        if st.button("Run cycle across universe", type="primary"):
            with st.spinner(f"Running {len(S.tickers)} tickers..."):
                st.session_state["cycle"] = run_all(
                    strategy_name=strat, qty=qty, wait_for_fill=wait, use_ml=use_ml)

        if st.button("Flatten all positions"):
            client = get_trading_client()
            client.close_all_positions(cancel_orders=True)
            st.warning("Close-all submitted.")

    with right:
        st.subheader("Positions")
        positions = get_trading_client().get_all_positions()
        if positions:
            pos_df = pd.DataFrame([{
                "Ticker": p.symbol,
                "Qty": float(p.qty),
                "Avg Entry": float(p.avg_entry_price),
                "Current": float(p.current_price),
                "Mkt Value": float(p.market_value),
                "Unreal P&L": float(p.unrealized_pl),
                "P&L %": float(p.unrealized_plpc),
            } for p in positions])
            st.dataframe(
                pos_df.style.format({
                    "Avg Entry": "${:,.2f}", "Current": "${:,.2f}",
                    "Mkt Value": "${:,.0f}", "Unreal P&L": "${:,.2f}", "P&L %": "{:.2%}"}),
                use_container_width=True, hide_index=True)
            st.metric("Total Unrealized P&L", f"${pos_df['Unreal P&L'].sum():,.2f}")
        else:
            st.info("No open positions.")

    if "cycle" in st.session_state:
        st.subheader("Last Cycle")
        st.dataframe(pd.DataFrame(st.session_state["cycle"]),
                     use_container_width=True, hide_index=True)

    st.subheader("Recent Orders")
    try:
        orders = OrderStore(ORDERS_DB).get_recent(25)
        st.dataframe(orders, use_container_width=True, hide_index=True) if not orders.empty \
            else st.info("No orders yet.")
    except Exception:
        st.info("No order history yet.")


# --- LIVE DATA: stream quotes, chart history ---
with tab_live:
    st.subheader("Live Quotes")
    symbols = st.multiselect("Stream", S.tickers, default=S.tickers[:3])

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Start stream") and symbols:
        streamer.start_stream(symbols, S.db_path)
        st.session_state["streaming"] = True
    if b2.button("Stop stream"):
        streamer.stop_stream()
        st.session_state["streaming"] = False

    running = streamer.is_running()
    st.caption(f"Stream: {'RUNNING' if running else 'STOPPED'} | "
               f"ticks stored: {QuoteStore(S.db_path).count() if os.path.exists(S.db_path) else 0}")

    if running:
        quotes = streamer.get_latest()
        cols = st.columns(max(len(quotes), 1))
        for col, (sym, q) in zip(cols, quotes.items()):
            col.markdown(f"**{sym}**")
            col.metric("Bid", q["bid"] if q["bid"] else "—")
            col.metric("Ask", q["ask"] if q["ask"] else "—")
            col.metric("Last", q["last"] if q["last"] else "—")
        if all(q["bid"] is None for q in quotes.values()):
            st.caption("Waiting for quotes — market open 9:30-16:00 ET, Mon-Fri.")

    st.divider()
    st.subheader("Historical")
    hc1, hc2 = st.columns([1, 3])
    ticker = hc1.selectbox("Ticker", S.tickers, key="hist_ticker")
    days = hc1.slider("Days", 30, 365, 90)

    if hc1.button("Load chart"):
        st.session_state["bars"] = fetch_bars(ticker, days=days)
        st.session_state["bars_ticker"] = ticker

    if "bars" in st.session_state:
        bars = st.session_state["bars"]
        hc2.plotly_chart(build_plot(bars), use_container_width=True)
        day = st.selectbox("Intraday detail", list(bars.index.date)[-30:])
        intraday = get_day_data(
            fetch_bars(st.session_state["bars_ticker"], days=30,
                       timeframe=TimeFrame(5, TimeFrameUnit.Minute)), day)
        if not intraday.empty:
            st.plotly_chart(build_plot(intraday), use_container_width=True)

    if running:
        time.sleep(1)
        st.rerun()


# --- BACKTEST ---
with tab_bt:
    bc1, bc2 = st.columns([1, 3])
    bt_ticker = bc1.selectbox("Ticker", S.tickers, key="bt_ticker")
    bt_days = bc1.slider("Lookback (days)", 180, 1200, 365)
    capital = bc1.number_input("Capital", 10_000, 1_000_000, 100_000, step=10_000)
    fee = bc1.number_input("Fee (bps)", 0.0, 50.0, 0.0)

    if bc1.button("Run backtest", type="primary"):
        with st.spinner("Backtesting..."):
            st.session_state["bt"] = run_all_strategies(bt_ticker, bt_days, capital, fee)

    if "bt" in st.session_state:
        results = st.session_state["bt"]
        equity = pd.DataFrame({k: v.equity for k, v in results.items()})
        bc2.line_chart(equity)
        st.dataframe(format_table(comparison_table(results)), use_container_width=True)


# --- CONFIG (read-only view of config.yaml) ---
with tab_cfg:
    st.subheader("Universe")
    st.write(", ".join(S.tickers))

    st.subheader("Risk Limits")
    st.table(pd.DataFrame([
        {"Limit": "Max shares / asset", "Value": f"{S.max_position_shares:g}"},
        {"Limit": "Max notional / order", "Value": f"${S.max_notional_per_order:,.0f}"},
        {"Limit": "Max portfolio exposure", "Value": f"{S.max_portfolio_exposure_pct:.0%} of equity"},
        {"Limit": "Stop loss", "Value": f"{S.stop_loss_pct:.0%}"},
        {"Limit": "Take profit", "Value": f"{S.take_profit_pct:.0%}"},
    ]))
    st.caption("Edit config.yaml and restart to change these.")
