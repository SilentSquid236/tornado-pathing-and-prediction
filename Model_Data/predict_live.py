import os
import sys
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

# Import your custom model
from model import TwoStreamTornadoPredictor

# ==========================================
# 🎛️ FORECASTER CONTROL PANEL
# ==========================================
# 🕒 TIME SELECTOR
# Next 12 hours: START = 1, END = 12
START_HOUR = 1
END_HOUR = 12

# 🗺️ REGION SELECTOR
# Options: "tornado_alley", "dixie_alley", "midwest", "northeast", "southeast", "custom"
TARGET_REGION = "midwest"

# Pre-defined Bounding Boxes: (West Lon, East Lon, South Lat, North Lat)
REGIONS = {
    "tornado_alley": (-105.0, -90.0, 30.0, 45.0),
    "dixie_alley":   (-95.0, -80.0, 30.0, 38.0),
    "midwest":       (-98.0, -80.0, 38.0, 49.0),
    "northeast":     (-82.0, -67.0, 38.0, 48.0),
    "southeast":     (-88.0, -75.0, 25.0, 35.0),
    "custom":        (-100.0, -90.0, 35.0, 40.0)
}
BOUNDING_BOX = REGIONS[TARGET_REGION]

# --- DIRECTORY SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "nam_output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "hrrr_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
# ==========================================

def get_optimal_hrrr_run(max_fxx):
    """Smart logic to find the latest valid HRRR run."""
    now = (datetime.utcnow() - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    
    if max_fxx <= 18:
        return now 
    else:
        valid_extended_hours = [0, 6, 12, 18]
        while now.hour not in valid_extended_hours:
            now -= timedelta(hours=1)
        return now

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
        H = Herbie(run_time, model='hrrr', product='prs', fxx=fxx, save_dir=CACHE_DIR, verbose=False)
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

def run_prediction(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Forecaster on: {device}")
    
    # Load Model
    model = TwoStreamTornadoPredictor(
        vars_2d=5, vars_3d=4, num_pressure_levels=39,
        grid_height=119, grid_width=144
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Setup the "Blank Canvas" for our Max Threat Composite
    max_prob_map = np.zeros((119, 144))
    saved_lats = None
    saved_lons = None

    run_time = get_optimal_hrrr_run(END_HOUR)
    start_valid_time = run_time + timedelta(hours=START_HOUR)
    end_valid_time = run_time + timedelta(hours=END_HOUR)
    
    print(f"Targeting NOAA HRRR Run: {run_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Scanning from +{START_HOUR}h to +{END_HOUR}h for region: {TARGET_REGION.upper()}...\n" + "-"*40)

    # THE TIME LOOP
    for fxx in range(START_HOUR, END_HOUR + 1):
        print(f"--> Processing Forecast Hour +{fxx}...")
        ds_sfc, ds_prs = get_hrrr_fxx(run_time, fxx)
        
        if ds_sfc is None or ds_prs is None:
            continue
            
        # Slice using the Region Selector dictionary
        sfc_win = slice_lonlat_box(ds_sfc, *BOUNDING_BOX)
        prs_win = slice_lonlat_box(ds_prs, *BOUNDING_BOX)

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
            print(f"    [!] Skipping hour +{fxx} due to missing variables: {e}")
            continue

        t2d_np = np.stack([np.squeeze(arr) for arr in [t2m, prmsl, sp, u10, v10]])
        t3d_np = np.stack([np.squeeze(arr) for arr in [t, gh, u, v]])
        
        t2d = torch.from_numpy(t2d_np).float().unsqueeze(0).to(device)
        t3d = torch.from_numpy(t3d_np).float().unsqueeze(0).to(device)

        # Interpolation and Normalization
        t2d = F.interpolate(t2d, size=(119, 144), mode='bilinear', align_corners=False)
        b, c, z, h, w = t3d.shape
        t3d = F.interpolate(t3d.view(b, c * z, h, w), size=(119, 144), mode='bilinear', align_corners=False).view(b, c, z, 119, 144) 

        t2d = (t2d - t2d.mean()) / (t2d.std() + 1e-6)
        t3d = (t3d - t3d.mean()) / (t3d.std() + 1e-6)

        num_levels = min(t3d.shape[2], 39) 
        t3d = t3d[:, :, :num_levels, :, :]

        # AI Prediction
        with torch.no_grad():
            logits = model(t2d, t3d)
            current_prob_map = torch.sigmoid(logits).squeeze().cpu().numpy() * 100
            
        # COMPOSITE LOGIC
        max_prob_map = np.maximum(max_prob_map, current_prob_map)

    print("-" * 40 + "\nTime loop complete! Applying edge mask...")

    # THE NAN ERASER (Fixes the border bleed)
    edge = 6 
    max_prob_map[:edge, :] = np.nan
    max_prob_map[-edge:, :] = np.nan
    max_prob_map[:, :edge] = np.nan
    max_prob_map[:, -edge:] = np.nan

    # --- MAP RENDERER ---
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.Mercator())
        
        # Uses the region selected in the Control Panel
        ax.set_extent(BOUNDING_BOX, crs=ccrs.PlateCarree())
        
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
        
        contourf = ax.contourf(
            lon_grid, lat_grid, max_prob_map, 
            levels=levels, colors=colors, alpha=0.6, transform=ccrs.PlateCarree()
        )
        
        label_levels = [2, 5, 10, 15, 30, 45, 60] 
        contour_lines = ax.contour(
            lon_grid, lat_grid, max_prob_map, 
            levels=label_levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree()
        )
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%1.0f%%', colors='black', use_clabeltext=True)
        
        cbar = plt.colorbar(contourf, orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label('Maximum Tornado Probability (%)', fontsize=12, fontweight='bold')
        
        start_str = start_valid_time.strftime('%b %d, %H:%M')
        end_str = end_valid_time.strftime('%b %d, %H:%M UTC')
        
        region_title = TARGET_REGION.replace('_', ' ').title()
        if START_HOUR == END_HOUR:
            plt.title(f"{region_title} Tornado Threat Prediction\nValid: {start_str} UTC", fontsize=16, fontweight='bold')
        else:
            plt.title(f"{region_title} Max Tornado Threat Corridor\nValid: {start_str} to {end_str}", fontsize=16, fontweight='bold')
        
        map_path = os.path.join(OUTPUT_DIR, "spc_threat_outlook.png")
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Success! Map saved to: {map_path}\n")
        
    except ImportError:
        print("\nMap skipped. Ensure matplotlib and cartopy are installed.")

if __name__ == "__main__":
    WEIGHTS = os.path.join(SCRIPT_DIR,"Model_Data", "training_data", "tornado_predictor_weights.pth")
    if os.path.exists(WEIGHTS):
        run_prediction(WEIGHTS)
    else:
        print(f"Missing weights file at: {WEIGHTS}")