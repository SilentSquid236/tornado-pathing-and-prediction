import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from pathlib import Path

class TornadoDataset(Dataset):
    # ---> UPDATED TO Model_Data <---
    def __init__(self, data_dir="Model_Data/netcdf_output"):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Could not find directory: {self.data_dir}")
            
        self.file_list = sorted(list(self.data_dir.glob("*.nc")))
        self.vars_2d = ["t2m", "prmsl", "sp", "u10", "v10"] 
        self.vars_3d = ["t", "gh", "u", "v"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 1. NetCDF Loading 
        ds = xr.open_dataset(file_path, engine="netcdf4")
        
        arrays_2d = []
        for var in self.vars_2d:
            if var in ds:
                arrays_2d.append(ds[var].squeeze().values)
            else:
                dummy_shape = ds[list(ds.data_vars)[0]].squeeze().shape[-2:]
                arrays_2d.append(np.zeros(dummy_shape))
        tensor_2d = torch.tensor(np.stack(arrays_2d), dtype=torch.float32)

        arrays_3d = []
        for var in self.vars_3d:
            if var in ds:
                data = ds[var].squeeze().values
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=0)
                arrays_3d.append(data)
            else:
                dummy_shape = ds[list(ds.data_vars)[0]].squeeze().shape
                if len(dummy_shape) == 2:
                    dummy_shape = (1, dummy_shape[0], dummy_shape[1])
                arrays_3d.append(np.zeros(dummy_shape))
        tensor_3d = torch.tensor(np.stack(arrays_3d), dtype=torch.float32)

        tensor_2d = torch.nan_to_num(tensor_2d, nan=0.0)
        tensor_3d = torch.nan_to_num(tensor_3d, nan=0.0)
        ds.close()

        grid_h, grid_w = tensor_2d.shape[1], tensor_2d.shape[2]

        # 2. Spatial Label Loading
        try:
            date_str = file_path.stem.split('_')[2] 
            # self.data_dir.parent perfectly resolves to "Model_Data" now
            mask_file = self.data_dir.parent / "target_masks" / f"{date_str}_mask.npy"
            
            if mask_file.exists():
                label_tensor = torch.from_numpy(np.load(mask_file))
                if label_tensor.shape[-2:] != (grid_h, grid_w):
                    label_tensor = torch.transpose(label_tensor, -1, -2)
            else:
                label_tensor = torch.zeros((1, grid_h, grid_w), dtype=torch.float32)
        except Exception:
            label_tensor = torch.zeros((1, grid_h, grid_w), dtype=torch.float32)

        return tensor_2d, tensor_3d, label_tensor