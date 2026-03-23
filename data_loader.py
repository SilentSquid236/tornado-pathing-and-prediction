import pandas as pd


NOAA_SPC_TORNADO_CSV_URL = "https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv"



def load_tornado_data(
    start_year: int = 1993,
    end_year: int = 2023,
    min_magnitude: int = 3,
    csv_url: str = NOAA_SPC_TORNADO_CSV_URL,
) -> pd.DataFrame:
    """Fetch and clean NOAA SPC tornado data for modeling."""
    df = pd.read_csv(csv_url)

    required_columns = {"yr", "mag", "len"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    filtered = df[
        (df["yr"] >= start_year)
        & (df["yr"] <= end_year)
        & (df["mag"] >= min_magnitude)
    ].copy()

    filtered["mag"] = pd.to_numeric(filtered["mag"], errors="coerce")
    filtered["len"] = pd.to_numeric(filtered["len"], errors="coerce")
    filtered = filtered.dropna(subset=["mag", "len"])

    return filtered
