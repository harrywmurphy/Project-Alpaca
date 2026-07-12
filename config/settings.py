# Single source of truth. Reads config.yaml. Secrets stay in .env.
import os
from dataclasses import dataclass
from functools import lru_cache

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


@dataclass(frozen=True)
class Settings:
    tickers: list
    db_path: str
    strategy: str
    default_qty: float
    max_position_shares: float
    max_notional_per_order: float
    max_portfolio_exposure_pct: float
    stop_loss_pct: float
    take_profit_pct: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    risk = cfg["risk"]

    db_path = cfg["db_path"]
    if not os.path.isabs(db_path):
        db_path = os.path.join(ROOT, db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    return Settings(
        tickers=list(cfg["tickers"]),
        db_path=db_path,
        strategy=cfg["strategy"],
        default_qty=float(cfg["default_qty"]),
        max_position_shares=float(risk["max_position_shares"]),
        max_notional_per_order=float(risk["max_notional_per_order"]),
        max_portfolio_exposure_pct=float(risk["max_portfolio_exposure_pct"]),
        stop_loss_pct=float(risk["stop_loss_pct"]),
        take_profit_pct=float(risk["take_profit_pct"]),
    )
