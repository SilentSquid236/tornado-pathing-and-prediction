import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
import torch

print("1. Waking up the GPU...")
device = torch.device("cuda")

try:
    print("2. Attempting to allocate memory...")
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    
    print("3. Attempting heavy matrix multiplication...")
    z = torch.matmul(x, y)
    
    print("4. Attempting a basic neural network layer (Convolution)...")
    conv = torch.nn.Conv2d(3, 16, kernel_size=3).to(device)
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    output = conv(dummy_image)
    
    print("\nSUCCESS! The RX 7600 survived basic training operations.")
    
except Exception as e:
    print(f"\nCRASHED with a Python error: {e}")