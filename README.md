# Project Alpaca — Systematic Trading System

End-to-end systematic trading system on Alpaca **paper trading only**.
Live data → signal → risk gate → order → monitoring.

## Architecture

```
config.yaml  ──> config/settings.py ──> every module (universe, params, risk limits)

data/                  strategy/              risk/          execution/
  connector.py           features.py            limits.py      order_manager.py
  historical.py          model.py    (rules)                   engine.py
  streamer.py            ml_model.py (RF+PCA)
  storage.py             backtest.py
  logger.py              metrics.py
                         pca_transform.py
                         pipeline.py
                                    ui/app.py  (command & control)
```

Flow: `historical/streamer` → `strategy` → `risk.check_order` → `order_manager` → Alpaca.
Ticks and order states persist to SQLite (`db/`); events to `logs/`; models to `artifacts/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste Alpaca PAPER keys
```

## Run

```bash
streamlit run ui/app.py                    # dashboard
python run_backtest.py --ticker SPY        # backtest mode
python train_models.py                     # train ML artifacts
python -m execution.engine                 # paper trading, rule-based
python -m execution.engine --ml            # paper trading, ML signal
pytest -q                                  # tests
```

## Strategy

Three rule-based strategies (`strategy/model.py`), all long-only 0/1 exposure:

| Strategy | Entry | Exit |
|---|---|---|
| Trend Following | MACD > signal **and** ADX > 25 | MACD < signal |
| Mean Reversion | RSI < 30 **and** close < lower Bollinger | RSI > 70 **and** close > upper band |
| Custom | close > EMA200, RSI > 40, CMF > 0 | close < EMA200 **or** RSI > 70 |

Plus an ML signal (`strategy/ml_model.py`): 8 indicator features → PCA (90% variance) →
Random Forest predicting next-day direction. Long if P(up) > 0.6.
Scaler and PCA are fit on the training slice only.

**Intuition:** trend-following harvests momentum autocorrelation; mean-reversion harvests
short-horizon overreaction. ADX and EMA200 act as regime filters so each rule only fires
where its premise holds.

## Risk Controls

Enforced in `risk/limits.py`, called before every order. Fails closed.

| Limit | Default |
|---|---|
| Max shares per asset | 50 |
| Max notional per order | $15,000 |
| Max portfolio exposure | 60% of equity |
| Stop loss | −5% |
| Take profit | +10% |

Set in `config.yaml`. Stop-loss/take-profit override the strategy signal on open positions.

## Data Pipeline

- Historical: REST OHLCV bars (`data/historical.py`)
- Live: WebSocket quotes + trades on a daemon thread (`data/streamer.py`), IEX feed
- Storage: every tick → SQLite `ticks` table; every order state → SQLite `orders` table
- Logging: rotating file + console (`logs/pipeline.log`)

## UI

`ui/app.py` (Streamlit). Four tabs:
- **Control** — positions, unrealized P&L, recent orders, run cycle, flatten all
- **Live Data** — start/stop stream, bid/ask/last, candlestick + intraday charts
- **Backtest** — all strategies vs buy-and-hold, equity curves, metrics table
- **Config** — active universe and risk limits

Status bar shows broker connection, mode (PAPER), equity, buying power.

## Metrics

Total return, CAGR, volatility, Sharpe, Sortino, max drawdown, win rate, trade count.

## Limitations

- Daily bars, long-only, market orders only
- Single chronological train/test split (no walk-forward)
- IEX feed only (not consolidated NBBO)
- One cycle per invocation — no persistent scheduler

## Video

TODO: link

## Contributors

TODO
