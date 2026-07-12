# Train + persist one RF artifact bundle per ticker. Run before `--ml`.
import argparse

from config.settings import get_settings
from strategy.ml_model import train_and_save

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--days", type=int, default=1200)
    args = p.parse_args()

    for t in (args.tickers or get_settings().tickers):
        try:
            r = train_and_save(t, days=args.days)
            print(f"{t}: acc={r['accuracy']:.3f} PCs={r['n_components']}")
        except Exception as e:
            print(f"{t}: FAILED -- {e}")
