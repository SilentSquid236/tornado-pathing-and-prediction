import torch
import xarray as xr
import numpy as np
from pathlib import Path
from model import TornadoPredictor

# --- 1. CONFIGURATION ---
# The exact stats your dataset generated earlier!
MEAN = torch.tensor([4.0852837562561035, 256.0007019042969]).view(-1, 1, 1)
STD = torch.tensor([2.5403850078582764, 79.48683166503906]).view(-1, 1, 1)

def normalize_tensor(tensor):
    return (tensor - MEAN) / (STD + 1e-7)

def predict_weather_map(nc_file_path, weights_path):
    # --- 2. LOAD THE BRAIN ---
    model = TornadoPredictor(num_channels=2)
    # Load the saved weights (map_location ensures it works even if you don't have a GPU)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    # model.eval() turns off training features like Dropout so it acts deterministically
    model.eval() 

    # --- 3. LOAD & PREP THE WEATHER MAP ---
    with xr.open_dataset(nc_file_path) as ds:
        wind = np.nan_to_num(ds["wind_speed"].squeeze().values, nan=0.0)
        temp = np.nan_to_num(ds["t2m"].squeeze().values, nan=0.0)
        
    stacked_data = np.stack([wind, temp], axis=0)
    tensor_x = torch.tensor(stacked_data, dtype=torch.float32)
    tensor_x = normalize_tensor(tensor_x)
    
    # The model expects a batch dimension, even for 1 file. 
    # unsqueeze(0) changes shape from (2, 144, 119) to (1, 2, 144, 119)
    tensor_x = tensor_x.unsqueeze(0)

    # --- 4. MAKE THE PREDICTION ---
    # torch.no_grad() saves memory since we aren't training/calculating gradients anymore
    with torch.no_grad():
        output = model(tensor_x)
        # Convert the raw decimal (e.g., 0.85) into a readable percentage (85.0%)
        probability = output.item() * 100

    print("\n" + "="*40)
    print(f"Analyzing File: {Path(nc_file_path).name}")
    print(f"Tornado Probability: {probability:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO TEST ---
    WEIGHTS_FILE = Path("Model Data/tornado_predictor_weights.pth")
    
    # Pick ANY single .nc file from your netcdf_output folder to test
    TEST_MAP = Path("Model Data/netcdf_output/tornado_alley_20160401_00z.nc")
    
    # Run the prediction
    if TEST_MAP.exists() and WEIGHTS_FILE.exists():
        predict_weather_map(TEST_MAP, WEIGHTS_FILE)
    else:
        print("Error: Could not find either the weights file or the test map. Check your paths!")