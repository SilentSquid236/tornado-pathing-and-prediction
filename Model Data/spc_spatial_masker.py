import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
SPC_URLS = [
    "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv",
    "https://www.spc.noaa.gov/wcm/data/1950-2022_actual_tornadoes.csv"
]
MIN_MAGNITUDE = 3
MIN_YEAR = 2016
LAT_MIN, LAT_MAX = 30.0, 45.0
LON_MIN, LON_MAX = -105.0, -90.0

GRID_WIDTH = 144  
GRID_HEIGHT = 119 

SCRIPT_DIR = Path(__file__).resolve().parent
# ---> UPDATED TO Model_Data <---
MASK_DIR = SCRIPT_DIR / "Model_Data" / "target_masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

def latlon_to_pixel(lat, lon):
    norm_x = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    norm_y = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    
    x_idx = int(norm_x * GRID_WIDTH)
    y_idx = int(norm_y * GRID_HEIGHT)
    
    x_idx = max(0, min(GRID_WIDTH - 1, x_idx))
    y_idx = max(0, min(GRID_HEIGHT - 1, y_idx))
    return y_idx, x_idx

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
        print("Error: Could not download the SPC database.")
        return

    local_ef3_plus = df[
        (df['mag'] >= MIN_MAGNITUDE) & (df['yr'] >= MIN_YEAR) &
        (df['slat'] >= LAT_MIN) & (df['slat'] <= LAT_MAX) &
        (df['slon'] >= LON_MIN) & (df['slon'] <= LON_MAX)
    ]
    
    grouped_by_date = local_ef3_plus.groupby('date')
    print(f"\nGenerating 2D Spatial Masks for {len(grouped_by_date)} tornadic days...")
    
    saved_count = 0
    for date_str, storm_group in grouped_by_date:
        formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        daily_mask = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        
        for _, storm in storm_group.iterrows():
            y, x = latlon_to_pixel(storm['slat'], storm['slon'])
            daily_mask[0, y, x] = 1.0  
            
            # Make the tornado hitbox slightly larger (3x3) for easier training
            if y > 0 and y < GRID_HEIGHT - 1 and x > 0 and x < GRID_WIDTH - 1:
                daily_mask[0, y-1:y+2, x-1:x+2] = 1.0
                
        np.save(MASK_DIR / f"{formatted_date}_mask.npy", daily_mask)
        saved_count += 1
        
    print(f"Success! Saved {saved_count} spatial target maps to: {MASK_DIR}")

if __name__ == "__main__":
    main()