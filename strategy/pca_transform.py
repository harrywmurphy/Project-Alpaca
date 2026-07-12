# Standardize -> PCA. Fit on TRAIN ONLY (else test stats leak into training).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VARIANCE_THRESHOLD = 0.90


def fit_pca(feature_df: pd.DataFrame, variance_threshold: float = VARIANCE_THRESHOLD):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)

    full = PCA().fit(scaled)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, variance_threshold) + 1)
    n = max(1, min(n, scaled.shape[1]))

    pca = PCA(n_components=n)
    values = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(values, columns=[f"PC{i+1}" for i in range(n)], index=feature_df.index)
    return pca_df, scaler, pca, full.explained_variance_ratio_


def transform_new(rows: pd.DataFrame, scaler: StandardScaler, pca: PCA) -> pd.DataFrame:
    values = pca.transform(scaler.transform(rows))
    return pd.DataFrame(values, columns=[f"PC{i+1}" for i in range(pca.n_components_)],
                        index=rows.index)


def plot_explained_variance(evr, threshold: float = VARIANCE_THRESHOLD):
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum) + 1), cum, marker="o")
    ax.axhline(threshold, color="red", linestyle="--", label=f"{threshold:.0%}")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
