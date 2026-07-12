# Backtest mode: all strategies on one ticker, writes table + figures to report/.
import argparse
from pathlib import Path

from strategy.metrics import (comparison_table, format_table,
                              plot_drawdown, plot_equity_curve)
from strategy.pipeline import run_all_strategies

REPORT = Path("report")
FIGS = REPORT / "figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--fee-bps", type=float, default=0.0)
    args = p.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)

    results = run_all_strategies(args.ticker, args.days, args.capital, args.fee_bps)
    table = comparison_table(results)

    print(f"\n=== {args.ticker} | {args.days}d ===")
    print(format_table(table).to_string())
    table.to_csv(REPORT / f"{args.ticker}_metrics.csv")

    plot_equity_curve(results).savefig(FIGS / f"{args.ticker}_equity.png", dpi=150)
    best = max((k for k in results if k != "Buy & Hold"),
               key=lambda k: results[k].equity.iloc[-1])
    plot_drawdown(results[best], best).savefig(FIGS / f"{args.ticker}_drawdown.png", dpi=150)
    results[best].trades.to_csv(REPORT / f"{args.ticker}_trades.csv", index=False)

    print(f"\nWrote report/{args.ticker}_metrics.csv and figures to {FIGS}/")
