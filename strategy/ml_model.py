# Random Forest on PCA components. Target = next-day close up.
# Features are the same indicators the rule-based strategies use.
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.historical import fetch_bars
from data.logger import get_logger
from strategy import features as ind
from strategy.pca_transform import fit_pca, transform_new

logger = get_logger(__name__)

SIGNAL_THRESHOLD = 0.6  # long if P(up) > this
FEATURE_COLUMNS = ["macd", "macd_signal", "macd_hist", "adx", "rsi",
                   "bb_pctb", "ema200_ratio", "cmf"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = Path(ROOT) / "artifacts"


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = ind.macd(df["close"])
    bb = ind.bollinger_bands(df["close"])
    width = (bb["upper"] - bb["lower"]).replace(0, np.nan)
    out = pd.DataFrame({
        "macd": m["macd"],
        "macd_signal": m["signal"],
        "macd_hist": m["hist"],
        "adx": ind.adx(df["high"], df["low"], df["close"]),
        "rsi": ind.rsi(df["close"]),
        "bb_pctb": (df["close"] - bb["lower"]) / width,
        "ema200_ratio": df["close"] / ind.ema(df["close"], 200) - 1,
        "cmf": ind.cmf(df["high"], df["low"], df["close"], df["volume"]),
    })
    return out.replace([np.inf, -np.inf], np.nan)  # inf survives dropna(); kill it here


def build_target(df: pd.DataFrame) -> pd.Series:
    nxt = df["close"].shift(-1) / df["close"] - 1
    return (nxt > 0).astype(float).where(nxt.notna()).rename("target")


def time_series_split(X, y, test_size: float = 0.2):
    # Chronological. Never shuffle time series.
    i = int(len(X) * (1 - test_size))
    return X.iloc[:i], X.iloc[i:], y.iloc[:i], y.iloc[i:]


def train_model(X_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,           # shallow: daily direction is low signal-to-noise
        min_samples_leaf=10,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info("Test accuracy: %.3f", acc)
    logger.info("\n%s", classification_report(y_test, preds, zero_division=0))
    return acc


def generate_signals(model, X: pd.DataFrame, threshold: float = SIGNAL_THRESHOLD) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"proba_up": proba,
                         "signal": (proba > threshold).astype(int)}, index=X.index)


def train_and_save(ticker: str, days: int = 1200, test_size: float = 0.2,
                   threshold: float = SIGNAL_THRESHOLD) -> dict:
    raw = fetch_bars(ticker, days=days)
    if raw.empty:
        raise ValueError(f"No bars for {ticker}")

    data = build_feature_matrix(raw)[FEATURE_COLUMNS].join(build_target(raw)).dropna()
    X, y = data[FEATURE_COLUMNS], data["target"]

    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size)

    pca_train, scaler, pca, evr = fit_pca(X_train)
    pca_test = transform_new(X_test, scaler, pca)

    model = train_model(pca_train, y_train)
    accuracy = evaluate_model(model, pca_test, y_test)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{ticker}_artifacts.joblib"
    joblib.dump({"model": model, "scaler": scaler, "pca": pca,
                 "feature_columns": FEATURE_COLUMNS, "threshold": threshold,
                 "explained_variance_ratio": evr, "accuracy": accuracy}, path)

    logger.info("Saved %s artifacts (accuracy=%.3f)", ticker, accuracy)
    return {"ticker": ticker, "accuracy": accuracy, "n_components": pca.n_components_,
            "path": str(path)}


def load_artifacts(ticker: str) -> dict:
    path = ARTIFACT_DIR / f"{ticker}_artifacts.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No artifacts at {path}. Run: python train_models.py")
    return joblib.load(path)


def predict_latest_signal(ticker: str, days: int = 400, bundle: dict = None):
    # Returns (signal, proba_up, bar_date). Never refits on live data.
    bundle = bundle or load_artifacts(ticker)
    model, scaler, pca = bundle["model"], bundle["scaler"], bundle["pca"]
    cols, threshold = bundle["feature_columns"], bundle.get("threshold", SIGNAL_THRESHOLD)

    raw = fetch_bars(ticker, days=days)
    if raw.empty:
        raise ValueError(f"No bars for {ticker}")

    feat = build_feature_matrix(raw)[cols].dropna()
    if feat.empty:
        raise ValueError(f"Not enough history for {ticker}")

    latest = transform_new(feat.iloc[[-1]], scaler, pca)
    proba_up = float(model.predict_proba(latest)[0, 1])
    return (1 if proba_up > threshold else 0), proba_up, feat.index[-1]
