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
