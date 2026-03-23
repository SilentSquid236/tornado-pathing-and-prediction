from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from data_loader import load_tornado_data
from model import (
    train_path_length_model,
    train_path_length_vs_width_model,
    train_path_width_model,
    width_strength_correlation,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_FIGURE_PATH = FIGURES_DIR / "tornado_magnitude_vs_length_and_tornado_width.png"
LENGTH_WIDTH_FIGURE_PATH = FIGURES_DIR / "tornado_path_length_vs_tornado_width.png"


def format_linear_equation(y_var: str, x_var: str, slope: float, intercept: float) -> str:
    """Human-readable y = m*x + b (handles negative intercept cleanly)."""
    sign = "+" if intercept >= 0 else "−"
    abs_b = abs(intercept)
    return f"{y_var} = {slope:.4g} * {x_var} {sign} {abs_b:.4g}"


def add_regression_equation_label(
    ax,
    y_var: str,
    x_var: str,
    slope: float,
    intercept: float,
    fontsize: int = 9,
) -> None:
    """Place regression equation in upper-right of axes (axes coordinates)."""
    eq = format_linear_equation(y_var, x_var, slope, intercept)
    ax.text(
        0.98,
        0.98,
        eq,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.92,
        },
    )


def describe_correlation(value: float) -> str:
    abs_val = abs(value)
    if abs_val < 0.2:
        return "very weak"
    if abs_val < 0.4:
        return "weak"
    if abs_val < 0.6:
        return "moderate"
    if abs_val < 0.8:
        return "strong"
    return "very strong"


def main() -> None:
    tornado_df = load_tornado_data()
    _, metrics = train_path_length_model(tornado_df)
    _, width_metrics = train_path_width_model(tornado_df)
    _, len_wid_metrics = train_path_length_vs_width_model(tornado_df)
    width_mag_corr = width_strength_correlation(tornado_df)

    print("Tornado Path Length Model (Linear Regression)")
    print(f"Rows used: {len(tornado_df)}")
    print(f"Slope: {metrics['slope']:.4f}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print(f"R^2: {metrics['r2']:.4f}")
    print("Tornado width vs magnitude (Linear Regression)")
    print(f"Slope (tornado width vs mag): {width_metrics['slope']:.4f}")
    print(f"Intercept: {width_metrics['intercept']:.4f}")
    print(f"R^2: {width_metrics['r2']:.4f}")
    print("Path length vs tornado width (Linear Regression)")
    print(f"Slope (len vs tornado width): {len_wid_metrics['slope']:.6f}")
    print(f"Intercept: {len_wid_metrics['intercept']:.4f}")
    print(f"R^2: {len_wid_metrics['r2']:.4f}")
    print(f"Correlation (mag vs len): {metrics['correlation']:.4f}")
    print(f"Correlation (mag vs tornado width): {width_mag_corr:.4f}")
    print(f"Correlation (len vs tornado width): {len_wid_metrics['correlation']:.4f}")
    print(
        f"Interpretation (mag vs len): {describe_correlation(metrics['correlation'])} positive relationship"
    )
    print(
        f"Interpretation (mag vs tornado width): {describe_correlation(width_mag_corr)} positive relationship"
    )
    print(
        f"Interpretation (len vs tornado width): {describe_correlation(len_wid_metrics['correlation'])} "
        f"{'positive' if len_wid_metrics['correlation'] >= 0 else 'negative'} relationship"
    )

    save_analysis_figure(
        tornado_df,
        slope=metrics["slope"],
        intercept=metrics["intercept"],
        wid_slope=width_metrics["slope"],
        wid_intercept=width_metrics["intercept"],
        path=DEFAULT_FIGURE_PATH,
    )
    print(f"Figure saved: {DEFAULT_FIGURE_PATH}")

    save_length_vs_width_figure(
        tornado_df,
        slope=len_wid_metrics["slope"],
        intercept=len_wid_metrics["intercept"],
        path=LENGTH_WIDTH_FIGURE_PATH,
    )
    print(f"Figure saved: {LENGTH_WIDTH_FIGURE_PATH}")


def save_length_vs_width_figure(
    tornado_df,
    slope: float,
    intercept: float,
    path: Path,
    dpi: int = 300,
) -> None:
    """Scatter: tornado width (x) vs path length (y) with linear fit."""
    path.parent.mkdir(parents=True, exist_ok=True)

    df = tornado_df[["len", "wid"]].copy()
    df["len"] = pd.to_numeric(df["len"], errors="coerce")
    df["wid"] = pd.to_numeric(df["wid"], errors="coerce")
    df = df.dropna(subset=["len", "wid"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["wid"], df["len"], alpha=0.35, s=14, edgecolors="none")
    x_line = np.linspace(df["wid"].min(), df["wid"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="C1", linewidth=2, label="Linear fit")
    ax.set_xlabel("Tornado width (yards)")
    ax.set_ylabel("Path length (miles)")
    ax.set_title("Path length vs tornado width")
    ax.legend(loc="upper left")
    add_regression_equation_label(ax, "len", "tornado width", slope, intercept)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_analysis_figure(
    tornado_df,
    slope: float,
    intercept: float,
    wid_slope: float,
    wid_intercept: float,
    path: Path,
    dpi: int = 300,
) -> None:
    """Scatter plots: magnitude vs path length and vs tornado width, each with a linear fit."""
    path.parent.mkdir(parents=True, exist_ok=True)

    mag = tornado_df["mag"]
    path_len = tornado_df["len"]

    fig, (ax_len, ax_wid) = plt.subplots(1, 2, figsize=(12, 5))

    ax_len.scatter(mag, path_len, alpha=0.35, s=12, edgecolors="none")
    x_line = np.linspace(mag.min(), mag.max(), 100)
    ax_len.plot(x_line, slope * x_line + intercept, color="C1", linewidth=2, label="Linear fit")
    ax_len.set_xlabel("Tornado magnitude (EF scale)")
    ax_len.set_ylabel("Path length (miles)")
    ax_len.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_len.set_title("Magnitude vs path length")
    ax_len.legend(loc="upper left")
    add_regression_equation_label(ax_len, "len", "mag", slope, intercept)
    ax_len.grid(True, alpha=0.3)

    df_wid = tornado_df[["mag", "wid"]].copy()
    df_wid["mag"] = pd.to_numeric(df_wid["mag"], errors="coerce")
    df_wid["wid"] = pd.to_numeric(df_wid["wid"], errors="coerce")
    df_wid = df_wid.dropna(subset=["mag", "wid"])
    ax_wid.scatter(df_wid["mag"], df_wid["wid"], alpha=0.35, s=12, edgecolors="none", color="C2")
    x_wid = np.linspace(df_wid["mag"].min(), df_wid["mag"].max(), 100)
    ax_wid.plot(
        x_wid,
        wid_slope * x_wid + wid_intercept,
        color="C3",
        linewidth=2,
        label="Linear fit",
    )
    ax_wid.legend(loc="upper left")
    add_regression_equation_label(ax_wid, "tornado width", "mag", wid_slope, wid_intercept)
    ax_wid.set_xlabel("Tornado magnitude (EF scale)")
    ax_wid.set_ylabel("Tornado width (yards)")
    ax_wid.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_wid.set_title("Magnitude vs tornado width")
    ax_wid.grid(True, alpha=0.3)

    fig.suptitle("NOAA SPC tornadoes (filtered window)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
