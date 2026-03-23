from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def train_path_length_model(
    tornado_df: pd.DataFrame,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a linear regression model:
    path length (len) ~ tornado magnitude (mag).
    """
    if tornado_df.empty:
        raise ValueError("Input tornado DataFrame is empty.")

    x = tornado_df[["mag"]].to_numpy()
    y = tornado_df["len"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)
    metrics = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, predictions)),
        "correlation": float(np.corrcoef(tornado_df["mag"], tornado_df["len"])[0, 1]),
    }

    return model, metrics


def train_path_width_model(
    tornado_df: pd.DataFrame,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a linear regression model:
    tornado width (wid) ~ tornado magnitude (mag).
    """
    if tornado_df.empty:
        raise ValueError("Input tornado DataFrame is empty.")

    df = tornado_df[["mag", "wid"]].copy()
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["wid"] = pd.to_numeric(df["wid"], errors="coerce")
    df = df.dropna(subset=["mag", "wid"])

    if df.empty:
        raise ValueError("No valid rows for tornado width regression.")

    x = df[["mag"]].to_numpy()
    y = df["wid"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)
    metrics = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, predictions)),
        "correlation": float(np.corrcoef(df["mag"], df["wid"])[0, 1]),
    }

    return model, metrics


def train_path_length_vs_width_model(
    tornado_df: pd.DataFrame,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train a linear regression model:
    path length (len) ~ tornado width (wid).
    """
    if tornado_df.empty:
        raise ValueError("Input tornado DataFrame is empty.")

    df = tornado_df[["len", "wid"]].copy()
    df["len"] = pd.to_numeric(df["len"], errors="coerce")
    df["wid"] = pd.to_numeric(df["wid"], errors="coerce")
    df = df.dropna(subset=["len", "wid"])

    if df.empty:
        raise ValueError("No valid rows for path length vs tornado width regression.")

    x = df[["wid"]].to_numpy()
    y = df["len"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)
    metrics = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, predictions)),
        "correlation": float(np.corrcoef(df["wid"], df["len"])[0, 1]),
    }

    return model, metrics


def width_strength_correlation(tornado_df: pd.DataFrame) -> float:
    """Return correlation between tornado width (wid) and strength (mag)."""
    required_columns = {"mag", "wid"}
    missing = required_columns - set(tornado_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for correlation: {sorted(missing)}")

    corr_df = tornado_df[["mag", "wid"]].copy()
    corr_df["mag"] = pd.to_numeric(corr_df["mag"], errors="coerce")
    corr_df["wid"] = pd.to_numeric(corr_df["wid"], errors="coerce")
    corr_df = corr_df.dropna(subset=["mag", "wid"])

    if corr_df.empty:
        raise ValueError("No valid rows available to compute width-strength correlation.")

    return float(np.corrcoef(corr_df["mag"], corr_df["wid"])[0, 1])
