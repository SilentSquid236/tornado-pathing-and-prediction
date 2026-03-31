import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- 1. CONFIGURATION ---
SPC_URLS = [
    "https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv",
    "https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv",
    "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
]

# MINIMUM TORNADO RATING (3 = EF3 and higher)
MIN_MAGNITUDE = 3

# EARLIEST YEAR (Must be 2016 or later to match the NAM 12km GDEX archive)
MIN_YEAR = 2016

# TORNADO ALLEY BOUNDING BOX
LAT_MIN = 30.0   
LAT_MAX = 45.0   
LON_MIN = -105.0  
LON_MAX = -90.0  

def generate_date_windows(ef3_dates):
    """Generates the 9-day window (-7 days to +1 day) for a list of target dates."""
    all_required_dates = set()
    target_tornadic_dates = set()

    for date_str in ef3_dates:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        target_tornadic_dates.add(event_date.strftime("%Y%m%d"))
        
        for days_offset in range(-7, 2): 
            current_date = event_date + timedelta(days=days_offset)
            all_required_dates.add(current_date.strftime("%Y%m%d"))

    return sorted(list(all_required_dates)), sorted(list(target_tornadic_dates))

def main():
    print("Connecting to NOAA Storm Prediction Center...")
    
    df = None
    for url in SPC_URLS:
        try:
            df = pd.read_csv(url)
            print(f"Successfully loaded database from {url}")
            break
        except Exception:
            continue
            
    if df is None:
        print("Error: Could not download the SPC database. Check your internet or the SPC website.")
        return

    # --- 2. FILTER THE DATABASE ---
    local_ef3_plus = df[
        (df['mag'] >= MIN_MAGNITUDE) & 
        (df['yr'] >= MIN_YEAR) &
        (df['slat'] >= LAT_MIN) & (df['slat'] <= LAT_MAX) &
        (df['slon'] >= LON_MIN) & (df['slon'] <= LON_MAX)
    ]
    
    unique_dates = local_ef3_plus['date'].unique().tolist()
    
    if not unique_dates:
        print(f"\nNo EF3+ events found in this bounding box since {MIN_YEAR}!")
        return
        
    print(f"\nFound {len(unique_dates)} distinct days with EF3+ tornadoes in Tornado Alley since {MIN_YEAR}!")
        
    # --- 3. GENERATE THE WINDOWS ---
    required_dates, tornadic_dates = generate_date_windows(unique_dates)
    
    # --- 4. SAVE THE FILES (Using Relative Paths) ---
    # This looks for "Model Data" inside the folder where you run the script
    model_data_dir = Path("Model Data")
    
    # Safety feature: Create the folder if it doesn't exist
    model_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the files with the newly requested names
    with open("simulator_dates.txt", "w") as f:
        for d in required_dates:
            f.write(f"{d}\n")
            
    with open("target_tornadic_dates.txt", "w") as f:
        for d in tornadic_dates:
            f.write(f"{d}\n")

    print("\n" + "="*40)
    print(f"Total simulator days to download (9-day windows): {len(required_dates)}")
    print(f"Files successfully generated in the '{model_data_dir}' directory!")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()