# Project Alpaca — Systematic Trading System

End-to-end systematic trading system on Alpaca **paper trading only**.
Live data → signal → risk gate → order → monitoring.



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







