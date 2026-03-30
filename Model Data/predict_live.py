import os
from pathlib import Path
from datetime import datetime, timedelta

# --- 1. AMD HARDWARE OVERRIDES (CRITICAL) ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["MIOPEN_DEBUG_DISABLE_CUBN"] = "1"
os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from model import TwoStreamTornadoPredictor

# --- 2. CONFIGURATION ---
CACHE_DIR = Path.home() / "hrrr_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TORNADO_ALLEY_BOX = (-105.0, -90.0, 30.0, 45.0)
FORECAST_HOURS = 12 # How many hours into the future to scan

def _first_existing_var(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in ds.variables or n in ds.coords or n in ds.data_vars:
            return n
    return None

def slice_lonlat_box(ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.Dataset:
    lat_name = _first_existing_var(ds, ("latitude", "lat", "LAT", "nav_lat"))
    lon_name = _first_existing_var(ds, ("longitude", "lon", "LON", "nav_lon"))
    
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    if np.any(lon > 180):
        lon = np.where(lon > 180, lon - 360, lon)

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return ds

    y_dim = ds[lat_name].dims[0] 
    x_dim = ds[lat_name].dims[1]

    y_slice = slice(y_indices.min(), y_indices.max() + 1)
    x_slice = slice(x_indices.min(), x_indices.max() + 1)
    
    sub_ds = ds.isel(**{y_dim: y_slice, x_dim: x_slice})
    sub_mask = mask[y_slice, x_slice]
    for var in sub_ds.data_vars:
        sub_ds[var] = sub_ds[var].where(sub_mask, 0.0)

    return sub_ds

def get_hrrr_fxx(run_time, fxx):
    """Downloads a specific forecast hour from a specific model run."""
    try:
        H = Herbie(run_time, model='hrrr', product='prs', fxx=fxx, save_dir=str(CACHE_DIR), verbose=False)
        search_str = ":(TMP:2 m.*|PRMSL|MSLMA|PRES:surface|UGRD:10 m.*|VGRD:10 m.*|TMP:[0-9]+ mb|HGT:[0-9]+ mb|UGRD:[0-9]+ mb|VGRD:[0-9]+ mb)"
        
        saved_file = H.download(search_str)
        datasets = cfgrib.open_datasets(str(saved_file))
        
        merged_sfc = xr.Dataset()
        merged_prs = xr.Dataset()
        
        sfc_keys = {"t2m", "prmsl", "mslma", "sp", "10u", "u10", "10v", "v10", "pres", "t2", "tmp"}
        prs_keys = {"t", "gh", "u", "v", "hgt", "tmp", "ugrd", "vgrd"}
        
        for ds in datasets:
            is_3d = 'isobaricInhPa' in ds.dims
            for var in ds.data_vars:
                if is_3d and var in prs_keys:
                    merged_prs[var] = ds[var].drop_vars(['step', 'valid_time', 'time'], errors='ignore').compute()
                elif not is_3d and var in sfc_keys:
                    merged_sfc[var] = ds[var].drop_vars(['heightAboveGround', 'step', 'valid_time', 'surface', 'time', 'meanSea'], errors='ignore').compute()
            ds.close()
            
        return merged_sfc, merged_prs
    except Exception as e:
        print(f"    [!] Error downloading Hour +{fxx}: {e}")
        return None, None

def get_safe_var(ds: xr.Dataset, possible_names: list) -> np.ndarray:
    for name in possible_names:
        if name in ds.data_vars or name in ds.coords:
            return ds[name].values
    raise ValueError(f"Could not find any of {possible_names}.")

def run_max_threat_composite(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing 12-Hour Max Threat Forecaster on: {device}")
    
    # 1. Load Model Once
    model = TwoStreamTornadoPredictor(
        vars_2d=5, vars_3d=4, num_pressure_levels=39, # Assuming 39 levels max
        grid_height=119, grid_width=144
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Setup the "Blank Canvas" for our Max Threat Composite
    max_prob_map = np.zeros((119, 144))
    saved_lats = None
    saved_lons = None

    # Get the most recent stable HRRR run
    run_time = (datetime.utcnow() - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    start_valid_time = run_time + timedelta(hours=1)
    end_valid_time = run_time + timedelta(hours=FORECAST_HOURS)
    
    print(f"Targeting NOAA HRRR Run: {run_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Building Threat Composite from +1h to +{FORECAST_HOURS}h...\n" + "-"*40)

    # 3. THE TIME LOOP
    for fxx in range(1, FORECAST_HOURS + 1):
        print(f"--> Processing Forecast Hour +{fxx}...")
        ds_sfc, ds_prs = get_hrrr_fxx(run_time, fxx)
        
        if ds_sfc is None or ds_prs is None:
            continue # Skip if NOAA dropped this specific hour
            
        sfc_win = slice_lonlat_box(ds_sfc, *TORNADO_ALLEY_BOX)
        prs_win = slice_lonlat_box(ds_prs, *TORNADO_ALLEY_BOX)

        # Save coordinates for the map renderer on the first successful pass
        if saved_lats is None:
            saved_lats = get_safe_var(sfc_win, ['latitude', 'lat'])
            saved_lons = get_safe_var(sfc_win, ['longitude', 'lon'])
            saved_lons = np.where(saved_lons > 180, saved_lons - 360, saved_lons)

        try:
            t2m   = get_safe_var(sfc_win, ['t2m', 't2', 'tmp', '2t'])
            prmsl = get_safe_var(sfc_win, ['prmsl', 'msl', 'mslma'])
            sp    = get_safe_var(sfc_win, ['sp', 'pres', 'surface'])
            u10   = get_safe_var(sfc_win, ['u10', '10u', 'u', 'ugrd'])
            v10   = get_safe_var(sfc_win, ['v10', '10v', 'v', 'vgrd'])
            
            t  = get_safe_var(prs_win, ['t', 'tmp'])
            gh = get_safe_var(prs_win, ['gh', 'hgt'])
            u  = get_safe_var(prs_win, ['u', 'ugrd'])
            v  = get_safe_var(prs_win, ['v', 'vgrd'])
        except ValueError as e:
            print(f"    [!] Skipping hour +{fxx} due to missing variables.")
            continue

        t2d_np = np.stack([np.squeeze(arr) for arr in [t2m, prmsl, sp, u10, v10]])
        t3d_np = np.stack([np.squeeze(arr) for arr in [t, gh, u, v]])
        
        t2d = torch.from_numpy(t2d_np).float().unsqueeze(0).to(device)
        t3d = torch.from_numpy(t3d_np).float().unsqueeze(0).to(device)

        # Interpolation to standard grid
        t2d = F.interpolate(t2d, size=(119, 144), mode='bilinear', align_corners=False)
        b, c, z, h, w = t3d.shape
        t3d = F.interpolate(t3d.view(b, c * z, h, w), size=(119, 144), mode='bilinear', align_corners=False).view(b, c, z, 119, 144) 

        # Normalization
        t2d = (t2d - t2d.mean()) / (t2d.std() + 1e-6)
        t3d = (t3d - t3d.mean()) / (t3d.std() + 1e-6)

        num_levels = min(t3d.shape[2], 39) 
        t3d = t3d[:, :, :num_levels, :, :]

        # AI Prediction for this specific hour
        with torch.no_grad():
            logits = model(t2d, t3d)
            current_prob_map = torch.sigmoid(logits).squeeze().cpu().numpy() * 100
            
        # THE COMPOSITE LOGIC: Compare this hour's map to our master map and keep the highest values!
        max_prob_map = np.maximum(max_prob_map, current_prob_map)

    print("-" * 40 + "\nAll 12 hours processed! Rendering composite map...")

    # --- 4. MAP RENDERER ---
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent(TORNADO_ALLEY_BOX, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax.add_feature(cfeature.OCEAN, facecolor='#b0d0e0')
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1.5)
        
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(saved_lons.min(), saved_lons.max(), 144),
            np.linspace(saved_lats.min(), saved_lats.max(), 119)
        )
        
        levels = [2, 5, 10, 15, 30, 45, 60, 100]
        colors = ['#008b00', '#8b4513', '#ffc800', '#ff0000', '#ff69b4', '#912cee', '#104e8b']
        
        # Draw colors using the MAX composite map
        contourf = ax.contourf(
            lon_grid, lat_grid, max_prob_map, 
            levels=levels, colors=colors, alpha=0.6, transform=ccrs.PlateCarree()
        )
        
        # Add the labeled lines
        label_levels = [2, 5, 10, 15, 30, 45, 60] 
        contour_lines = ax.contour(
            lon_grid, lat_grid, max_prob_map, 
            levels=label_levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree()
        )
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%1.0f%%', colors='black', use_clabeltext=True)
        
        cbar = plt.colorbar(contourf, orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label('Maximum Tornado Probability (%)', fontsize=12, fontweight='bold')
        
        # Update Title to show the time window
        start_str = start_valid_time.strftime('%b %d, %H:%M')
        end_str = end_valid_time.strftime('%b %d, %H:%M UTC')
        plt.title(f"12-Hour Max Tornado Threat Corridor\nValid: {start_str} to {end_str}", fontsize=16, fontweight='bold')
        
        map_path = Path.cwd() / "spc_12hr_max_threat.png"
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Success! Composite Map saved to: {map_path}\n")
        
    except ImportError:
        print("\nMap skipped. Ensure matplotlib and cartopy are installed.")

if __name__ == "__main__":
    WEIGHTS = "Model_Data/training_data/tornado_predictor_weights.pth"
    if Path(WEIGHTS).exists():
        run_max_threat_composite(WEIGHTS)
    else:
        print(f"Missing weights file at: {WEIGHTS}")