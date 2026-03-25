"""
Packages the WRF NetCDF weather data for use in the PyTorch model training process.
Generates input tensors (X) and tornado classification labels (Y).
"""

import re
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("Model Data/Model Data/netcdf_output")
TARGET_VARS = ["wind_speed", "t2m"] 

class TornadoDataset(Dataset):
    def __init__(self, data_dir, variables, target_dates_file, transform=None):
        """
        Custom PyTorch Dataset for loading weather NetCDF files and generating labels.
        """
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.transform = transform
        
        # 1. Load the Answer Key (Master list of tornadic dates)
        self.target_dates = set()
        with open(target_dates_file, 'r') as f:
            for line in f:
                # Strip whitespace and add the YYYYMMDD string to our set
                self.target_dates.add(line.strip())
        
        # 2. Grab all the .nc files
        self.files = sorted(list(self.data_dir.glob("*.nc")))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No .nc files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # --- PART A: BUILD THE INPUT MAPS (X) ---
        with xr.open_dataset(file_path) as ds:
            channels = []
            for var in self.variables:
                if var in ds.data_vars:
                    # squeeze() removes the empty 'time' dimension so we just get a 2D grid
                    data_2d = ds[var].squeeze().values
                    # Replace any missing data (NaNs) with 0
                    data_2d = np.nan_to_num(data_2d, nan=0.0) 
                    channels.append(data_2d)
                else:
                    raise ValueError(f"Variable '{var}' not found in {file_path.name}")
            
            # Stack the 2D arrays into a 3D block: Shape becomes (Channels, Height, Width)
            stacked_data = np.stack(channels, axis=0)
            tensor_x = torch.tensor(stacked_data, dtype=torch.float32)
            
            if self.transform:
                tensor_x = self.transform(tensor_x)
                
        # --- PART B: BUILD THE LABEL (Y) ---
        # Search the filename for an 8-digit date string (YYYYMMDD)
        date_match = re.search(r'\d{8}', file_path.name)
        
        if date_match:
            file_date = date_match.group(0)
            # If this date is in our master list, it's a Tornado Day (1.0). Else, (0.0).
            is_tornado_day = 1.0 if file_date in self.target_dates else 0.0
        else:
            is_tornado_day = 0.0 
            
        tensor_y = torch.tensor([is_tornado_day], dtype=torch.float32)
        
        # Return both the weather map AND the answer
        return tensor_x, tensor_y


def get_dataset_stats(dataset):
    """
    Calculates Mean and Std Dev iteratively to prevent RAM crashes.
    Unpacks both X (maps) and Y (labels).
    """
    print("Iteratively scanning dataset for normalization stats...")
    
    num_channels = len(dataset.variables)
    channel_sum = torch.zeros(num_channels)
    channel_sum_sq = torch.zeros(num_channels)
    pixel_count = 0
    
    for i in range(len(dataset)):
        # Unpack the tuple! We only want to do math on tensor_x
        tensor_x, tensor_y = dataset[i] 
        
        # Sum across the Height and Width (dims 1 and 2)
        channel_sum += tensor_x.sum(dim=[1, 2])
        channel_sum_sq += (tensor_x ** 2).sum(dim=[1, 2])
        pixel_count += tensor_x.shape[1] * tensor_x.shape[2]
        
    mean = channel_sum / pixel_count
    var = (channel_sum_sq / pixel_count) - (mean ** 2)
    std = torch.sqrt(var)
    
    return mean.view(-1, 1, 1), std.view(-1, 1, 1)


def main():
    # 1. Use the Absolute Path for your answer key
    # Note: If your txt file is inside the 'Model Data' folder, add that to the path below!
    TARGET_DATES_FILE = Path("Model Data/target_tornadic_dates.txt")
    
    # Check if the file actually exists before doing anything else
    if not TARGET_DATES_FILE.exists():
        print(f"CRITICAL ERROR: Cannot find the text file at {TARGET_DATES_FILE}")
        print("Please check the exact file path and update TARGET_DATES_FILE in the script.")
        return

    # 2. Initialize the raw dataset
    raw_dataset = TornadoDataset(DATA_DIR, TARGET_VARS, TARGET_DATES_FILE)
    print(f"Successfully loaded dataset with {len(raw_dataset)} weather maps.")
    
    # 3. Calculate normalization statistics
    mean, std = get_dataset_stats(raw_dataset)
    print(f"Dataset Mean (per channel): {mean.squeeze().tolist()}")
    print(f"Dataset Std  (per channel): {std.squeeze().tolist()}")
    
    # 4. Create the normalizer function
    def normalize_tensor(tensor):
        return (tensor - mean.squeeze(0)) / (std.squeeze(0) + 1e-7)
    
    # 5. Create the final ML dataset with the transform applied
    ml_dataset = TornadoDataset(DATA_DIR, TARGET_VARS, TARGET_DATES_FILE, transform=normalize_tensor)
    
    # 6. Spin up the DataLoader (The Conveyor Belt)
    dataloader = DataLoader(ml_dataset, batch_size=2, shuffle=True)
    
    print("\nTesting the DataLoader conveyor belt...")
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  -> Maps (X) Shape: {batch_x.shape}")
        print(f"  -> Labels (Y) Shape: {batch_y.shape}")
        print(f"  -> Labels in this batch: {batch_y.squeeze().tolist()}")
        break # Just test the first batch

if __name__ == "__main__":
    main()