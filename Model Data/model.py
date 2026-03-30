import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStreamTornadoPredictor(nn.Module):
    def __init__(self, vars_2d, vars_3d, num_pressure_levels, grid_height, grid_width):
        super(TwoStreamTornadoPredictor, self).__init__()
        
        # 2D Stream (Surface)
        self.conv2d_1 = nn.Conv2d(vars_2d, 16, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(16)
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(32)
        
        # 3D Stream (Upper Air)
        self.conv3d_1 = nn.Conv3d(vars_3d, 16, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm3d(16)
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3d_2 = nn.BatchNorm3d(32)
        
        # Final Spatial Map Output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_2d, x_3d):
        x2 = F.relu(self.bn2d_1(self.conv2d_1(x_2d)))
        x2 = F.relu(self.bn2d_2(self.conv2d_2(x2)))
        
        x3 = F.relu(self.bn3d_1(self.conv3d_1(x_3d)))
        x3 = F.relu(self.bn3d_2(self.conv3d_2(x3)))
        
        # Collapse the Z-axis to merge with the 2D surface map
        x3, _ = torch.max(x3, dim=2) 
        
        x = torch.cat((x2, x3), dim=1)
        
        return self.final_conv(x)