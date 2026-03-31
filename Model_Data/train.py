import os
import gc

# --- 1. THE AMD HARDWARE FIXES (CRITICAL) ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["MIOPEN_DEBUG_DISABLE_CUBN"] = "1"
os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Disable MIOpen optimization inside Python as a secondary safety net
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Import your custom classes
from dataset_loader import TornadoDataset
from model import TwoStreamTornadoPredictor

def main():
    # --- 2. SETUP HARDWARE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

# --- 3. SETUP DATA ---
    # Dynamically find the exact folder this train.py script is sitting in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Point it directly to the "nam output" folder right next to the script
    data_path = os.path.join(script_dir, "nam_output")
    print(f"Looking for data in: {data_path}")
    
    tornado_dataset = TornadoDataset(data_dir=data_path)
    
    if len(tornado_dataset) == 0:
        print("Error: No data found! Check your folder paths.")
        return

    # ---> THIS IS THE LINE I ACCIDENTALLY DELETED! <---
    sample_2d, sample_3d, _ = tornado_dataset[0]
    
    print(f"\nAuto-detected Grid: {sample_2d.shape[1]}x{sample_2d.shape[2]} | Z-Levels: {sample_3d.shape[1]}")

    # DATALOADER: batch_size=1 is required for 8GB VRAM with 3D weather data
    dataloader = DataLoader(
        tornado_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )

    # --- 4. SETUP MODEL & OPTIMIZER ---
    model = TwoStreamTornadoPredictor(
        vars_2d=sample_2d.shape[0], 
        vars_3d=sample_3d.shape[0], 
        num_pressure_levels=sample_3d.shape[1], 
        grid_height=sample_2d.shape[1], 
        grid_width=sample_2d.shape[2]
    ).to(device)
    
    # ---> THE CLASS IMBALANCE FIX <---
    # Calculate the exact ratio of empty sky to tornado pixels
    total_pixels = sample_2d.shape[1] * sample_2d.shape[2]  # e.g., 119 * 144 = 17,136
    positive_pixels = 9.0  # Your 3x3 tornado hitbox
    negative_pixels = total_pixels - positive_pixels
    # 1. DEFINE the raw multiplier first
    raw_multiplier = negative_pixels / positive_pixels
    
    # 2. CAP it at 50.0 so the AI doesn't panic
    weight_multiplier = min(raw_multiplier, 50.0)
    
    print(f"Applying Class Imbalance Fix: Multiplying tornado pixel loss by {weight_multiplier:.1f}x")
    
    # Send the weight to the GPU
    pos_weight = torch.tensor([weight_multiplier]).to(device)
    
    # Initialize loss with the massive penalty for missing tornadoes
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scaler = torch.amp.GradScaler('cuda')

    # --- 5. THE TRAINING LOOP ---
    epochs = 5
    print("\nStarting Spatial Training (Map Generation Mode)...\n" + "="*30)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        processed_count = 0
        
        for batch_idx, (t2d, t3d, labels) in enumerate(dataloader):
            # Move data to RX 7600
            t2d = t2d.to(device).float()
            t3d = t3d.to(device).float()
            labels = labels.to(device).float()
            
            # Normalize on the fly
            t2d = (t2d - t2d.mean()) / (t2d.std() + 1e-6)
            t3d = (t3d - t3d.mean()) / (t3d.std() + 1e-6)

            optimizer.zero_grad(set_to_none=True)
            
            # Forward Pass with 16-bit Math
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(t2d, t3d)
                loss = criterion(predictions, labels)
            
            # Safety check: skip if the NetCDF file was corrupted
            if torch.isnan(loss):
                print(f"  Batch [{batch_idx+1}] skipped: Data NaN detected.")
                continue

            # Backward Pass
            scaler.scale(loss).backward()
            
            # Gradient Clipping prevents the math from exploding
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            processed_count += 1
            
            # Progress Update
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                avg_so_far = running_loss / processed_count if processed_count > 0 else 0
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] | Avg Loss: {avg_so_far:.4f}")
            
            # Aggressive VRAM Cleanup
            del t2d, t3d, labels, predictions, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        final_avg = running_loss / processed_count if processed_count > 0 else 0
        print(f"\n✅ Epoch [{epoch+1}/{epochs}] | Final Avg Loss: {final_avg:.4f}\n" + "-"*30)

    # --- 6. SAVE RESULTS ---
    save_path = "Model_Data/training_data/tornado_predictor_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Saved to: {save_path}")

if __name__ == "__main__":
    main()