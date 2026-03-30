#!/usr/bin/env python3
"""
Automated NAM 12km (GDEX d609000) downloader & processor for Tornado Alley.
HYBRID APPROACH: 
- Pure Numpy for bulletproof spatial slicing.
- Dynamic 2D/3D severe weather parameter extraction (CAPE, SRH, LFC, etc.).
- Calculates 0-500mb Bulk Wind Shear natively (No MetPy required).
- True Multi-threading: Downloads concurrently, reads safely.
"""

from __future__ import annotations

import os
import warnings

# --- WARNING SUPPRESSION ---
os.environ["PROJ_NETWORK"] = "OFF"
warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path")
warnings.filterwarnings("ignore", module="pyproj")

import threading
import concurrent.futures
import datetime
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

# --- CONFIGURATION ---
GRIB_LOCK = threading.Lock()
SCRIPT_DIR = Path(__file__).resolve().parent

# Safely pointing to the underscore directory
DATES_FILE = Path("Model_Data/simulator_dates.txt")
HOURS = ["00", "12"]

OUTPUT_DIR = SCRIPT_DIR / "Model_Data" / "netcdf_output"
WORK_DIR = SCRIPT_DIR / "Model_Data" / "_nam_work"

RDA_GDEX_NAM_BASE = "https://data.rda.ucar.edu/d609000"
OSDF_GDEX_NAM_BASE = "https://osdf-director.osg-htc.org/ncar/gdex/d609000"

# Tornado Alley bounding box
TORNADO_ALLEY_BOX = (-105.0, -90.0, 30.0, 45.0)
DOWNLOAD_USER_AGENT = "tornado-pathing-bot/1.0 (research)"

# ----------------------------------------------------------------------------------

def nam_grib_basename_variants(ymd: str, hour_utc: str) -> list[str]:
    base_with_date = f"{ymd}.nam.t{hour_utc}z.awphys00"
    base_no_date = f"nam.t{hour_utc}z.awphys00"
    
    return [
        f"{base_with_date}.grb2.tm00",
        f"{base_with_date}.grib2.tm00",
        f"{base_no_date}.grb2.tm00",   
        f"{base_no_date}.grib2.tm00",
        f"{base_with_date}.tm00.grb2",
        f"{base_with_date}.tm00.grib2",
        f"{base_no_date}.tm00.grb2",
        f"{base_no_date}.tm00.grib2"
    ]

def nam_download_candidate_urls(year: str, ymd: str, hour_utc: str) -> list[str]:
    filenames = nam_grib_basename_variants(ymd, hour_utc)
    ym = ymd[:6] 
    
    bases = [
        "https://data.rda.ucar.edu/d609000",
        "https://osdf-director.osg-htc.org/ncar/gdex/d609000",
        "https://data.rda.ucar.edu/ds609.0",      
        "https://gdex.ucar.edu/datasets/d609000"  
    ]
    
    folder_patterns = [
        f"{year}/{ymd}",  
        f"{year}/{ym}",   
        f"{year}",        
    ]
    
    urls = []
    for base in bases:
        for pattern in folder_patterns:
            for fn in filenames:
                urls.append(f"{base}/{pattern}/{fn}")
                
    return urls

def _first_existing_var(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in ds.variables or n in ds.coords or n in ds.data_vars:
            return n
    return None

def download_grib(url: str, dest: Path, timeout_s: int = 900) -> None:
    dest = Path(dest).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": DOWNLOAD_USER_AGENT}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            with open(dest, "wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Download failed: HTTP {e.code} {e.reason} for URL: {url}") from e

def download_first_working_url(urls: list[str], dest: Path, timeout_s: int = 900) -> str:
    dest = Path(dest).expanduser().resolve()
    last_err: Exception | None = None
    for url in urls:
        try:
            download_grib(url, dest, timeout_s=timeout_s)
            size = dest.stat().st_size
            
            if size < 100 * 1024:
                raise RuntimeError(f"Download only {size/1024:.1f} KB; likely an HTML error page.")
                
            with open(dest, "rb") as fh:
                if fh.read(4) != b"GRIB":
                    raise RuntimeError("File does not start with GRIB magic.")
            return url
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink(missing_ok=True)
    raise RuntimeError(f"None of {len(urls)} URLs worked. Last error: {last_err}") from last_err

def open_nam_grib_dynamic(path: Path) -> tuple[xr.Dataset, dict]:
    try:
        datasets = cfgrib.open_datasets(str(path))
    except Exception as e:
        raise RuntimeError(f"cfgrib failed to parse the file completely: {e}")

    if not datasets:
        raise RuntimeError(f"cfgrib could not extract any datasets from {path.name}")

    vars_2d = {
        "t2m", "prmsl", "sp", "10u", "u10", "10v", "v10", "2t", "TMP", "UGRD", "VGRD",
        "cape", "cin", "hlcy", "2d", "dpt", "pwat", "lcl", "hlfc", "plfc"
    }
    vars_3d = {"t", "u", "v", "gh", "w", "absv"}
    
    merged_ds = xr.Dataset()
    found_vars = []

    for ds in datasets:
        is_3d_cube = 'isobaricInhPa' in ds.dims
        
        for var in ds.data_vars:
            if var in vars_3d and is_3d_cube:
                if var not in merged_ds:
                    clean_da = ds[var].drop_vars(['step', 'valid_time', 'time'], errors="ignore")
                    merged_ds[var] = clean_da.compute()  
                    found_vars.append(var)
                    
            elif var in vars_2d and not is_3d_cube:
                if var not in merged_ds:
                    clean_da = ds[var].drop_vars(['heightAboveGround', 'step', 'valid_time', 'surface', 'time', 'meanSea'], errors="ignore")
                    merged_ds[var] = clean_da.compute()  
                    found_vars.append(var)
            
        ds.close() 

    if not found_vars:
        return datasets[0], {"fallback": "first_dataset"}
    
    return merged_ds, {"extracted_vars": found_vars}

def slice_lonlat_box(ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.Dataset:
    coords_to_drop = [c for c in ["step", "valid_time", "surface", "heightAboveGround"] if c in ds.coords]
    ds = ds.drop_vars(coords_to_drop, errors="ignore")
    
    if "time" in ds.dims and ds.sizes.get("time", 0) >= 1:
        ds = ds.squeeze("time", drop=True)

    lat_name = _first_existing_var(ds, ("latitude", "lat", "LAT", "nav_lat"))
    lon_name = _first_existing_var(ds, ("longitude", "lon", "LON", "nav_lon"))
    
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    
    if np.any(lon > 180):
        lon = np.where(lon > 180, lon - 360, lon)

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    y_indices, x_indices = np.where(mask)

    y_dim = ds[lat_name].dims[0] 
    x_dim = ds[lat_name].dims[1]

    y_slice = slice(y_indices.min(), y_indices.max() + 1)
    x_slice = slice(x_indices.min(), x_indices.max() + 1)
    
    sub_ds = ds.isel(**{y_dim: y_slice, x_dim: x_slice})
    
    sub_mask = mask[y_slice, x_slice]
    for var in sub_ds.data_vars:
        sub_ds[var] = sub_ds[var].where(sub_mask, 0.0)

    return sub_ds

def add_severe_weather_params(ds: xr.Dataset) -> xr.Dataset:
    u_sfc_name = _first_existing_var(ds, ("10u", "u10", "UGRD", "u_sfc"))
    v_sfc_name = _first_existing_var(ds, ("10v", "v10", "VGRD", "v_sfc"))

    if u_sfc_name and v_sfc_name:
        u_sfc = ds[u_sfc_name]
        v_sfc = ds[v_sfc_name]
        
        # 1. Calculate general surface wind speed natively
        ds["surface_wind_speed"] = np.sqrt(u_sfc**2 + v_sfc**2)

        # 2. Calculate 0-500mb Bulk Wind Shear natively
        if "u" in ds and "v" in ds and "isobaricInhPa" in ds.coords:
            try:
                u_500 = ds["u"].sel(isobaricInhPa=500.0)
                v_500 = ds["v"].sel(isobaricInhPa=500.0)
                
                shear_u = u_500 - u_sfc
                shear_v = v_500 - v_sfc
                
                ds["bulk_shear_500mb"] = np.sqrt(shear_u**2 + shear_v**2)
            except KeyError:
                pass
                
    return ds

def cleanup_grib(path: Path) -> None:
    if path.exists():
        path.unlink()
    for idx_file in path.parent.glob(f"{path.name}*.idx"):
        idx_file.unlink(missing_ok=True)

def process_one_grib(candidate_urls: list[str], filename: str, ymd: str, hr: str) -> Path:
    temp_grib = WORK_DIR / filename
    ds: xr.Dataset | None = None
    try:
        # 1. NO LOCK HERE: All workers download massive files at the same time
        used_url = download_first_working_url(candidate_urls, temp_grib)
        
        # 2. LOCK HERE: Workers take turns asking the fragile C-library to open the file
        with GRIB_LOCK:
            ds, used_filter = open_nam_grib_dynamic(temp_grib)
        
        # 3. NO LOCK HERE: Workers go back to doing independent math simultaneously
        sub = slice_lonlat_box(ds, *TORNADO_ALLEY_BOX)
        sub = add_severe_weather_params(sub)
        
        sub.attrs["source_url"] = used_url
        sub.attrs["source_file"] = filename
        sub.attrs["cfgrib_filter"] = json.dumps(used_filter)
        
        valid_time = pd.to_datetime(f"{ymd} {hr}:00:00")
        sub = sub.expand_dims(time=[valid_time])
        
        for var in sub.variables:
            if '_metpy_axis' in sub[var].attrs:
                del sub[var].attrs['_metpy_axis']
                
        output_nc = OUTPUT_DIR / f"tornado_alley_{ymd}_{hr}z.nc"
        sub.to_netcdf(output_nc)
        
        return output_nc
    finally:
        if ds is not None:
            ds.close()
        cleanup_grib(temp_grib)

def process_one_grib_safe(args):
    candidate_urls, filename, ymd, hr = args
    return process_one_grib(candidate_urls, filename, ymd, hr)

def main() -> int:
    print(f"Starting NAM Pipeline with True Multi-Threading (5GHz Optimized).")
    
    if not DATES_FILE.exists():
        print(f"CRITICAL ERROR: Cannot find {DATES_FILE}")
        return 1
        
    with open(DATES_FILE, 'r') as f:
        target_dates = [line.strip() for line in f if line.strip()]
        
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for ymd in target_dates:
        year = ymd[:4] 
        for hr in HOURS:
            filename = f"temp_{ymd}_{hr}z.grib"
            candidates = nam_download_candidate_urls(year, ymd, hr)
            tasks.append((candidates, filename, ymd, hr))

    successes = 0
    failures = 0

    # 4 workers is a great sweet spot for 16GB RAM and a 5GHz CPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(process_one_grib_safe, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            _, filename, _, _ = future_to_task[future]
            try:
                out_path = future.result()
                successes += 1
                print(f"[SUCCESS] {out_path.name}")
            except Exception as e:
                failures += 1
                print(f"[FAILED] {filename}: {e}")

    print(f"\nFinished. Success: {successes}, Fail: {failures}")
    return 0 if successes > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())