import xarray as xr
from pathlib import Path

def inspect_weather_data():
    data_dir = Path("Model Data/netcdf_output")
    files = list(data_dir.glob("*.nc"))
    
    if not files:
        print("No NetCDF files found in netcdf_output!")
        return
        
    # Just grab the very first file to peek inside
    file_path = files[0]
    print(f"Inspecting File: {file_path.name}\n" + "-"*30)
    
    ds = xr.open_dataset(file_path, engine="netcdf4")
    
    print("1. Coordinates (The Grid & Altitude):")
    for coord in ds.coords:
        # We are looking for something like 'isobaricInhPa'
        print(f"   -> {coord}: {ds[coord].values}")
        
    print("\n2. Upper Air Variables (The 3D Data):")
    for var in ['t', 'u', 'v', 'gh']:
        if var in ds:
            print(f"   -> '{var}' shape: {ds[var].shape}")
        else:
            print(f"   -> '{var}' is MISSING from this file!")
            
    ds.close()

if __name__ == "__main__":
    inspect_weather_data()