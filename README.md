# Project Alpaca — Systematic Trading System

Live, systematic trading system built with Alpaca's trading API. We created 3 trading strategies along with an ML Random Forest signal and created an engine to deploy them in Alpaca's paper trading environment. Our system also collects live and historical bid/ask data, and the data collection allowed us to create a backtester to gauge the performance of our signals historically. Going into this project, we were hoping to develop a feel for how high frequency traders and quantitative traders build an environment to create, test, and deploy their trading strategies. 

## VIDEO WALKTHROUGH
LINK: https://drive.google.com/file/d/1mOUPl-I5TUzaisU56DYybi8J88PiYNBP/view?usp=sharing

## Setup

```bash
git clone https://github.com/harrywmurphy/Project-Alpaca.git
cd project-alpaca
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Create `.env` in the repo root 

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

| Trend Following | MACD > signal **and** ADX > 25 | MACD < signal |
| Mean Reversion | RSI < 30 **and** close < lower Bollinger | RSI > 70 **and** close > upper band |
| Custom | close > EMA200, RSI > 40, CMF > 0 | close < EMA200 **or** RSI > 70 |

Plus an ML signal (`strategy/ml_model.py`): 8 indicator features → PCA (90% variance) →
Random Forest predicting next-day direction. Long if P(up) > 0.6.

## Example usage

Launch the dashboard and confirm the status bar shows Connected and PAPER. Select a strategy on the Control tab, set the order quantity, and run a cycle across the universe. Signals, orders, fills, positions, and P&L populate in the tables below. The Live Data tab streams quotes, the Backtest tab runs historical simulations, and the Config tab shows the active tickers and risk limits.





