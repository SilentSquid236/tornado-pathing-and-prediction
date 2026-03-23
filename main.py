from data_loader import load_tornado_data
from model import train_path_length_model


def main() -> None:
    tornado_df = load_tornado_data()
    _, metrics = train_path_length_model(tornado_df)

    print("Tornado Path Length Model (Linear Regression)")
    print(f"Rows used: {len(tornado_df)}")
    print(f"Slope: {metrics['slope']:.4f}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print(f"R^2: {metrics['r2']:.4f}")
    print(f"Correlation (mag vs len): {metrics['correlation']:.4f}")


if __name__ == "__main__":
    main()
