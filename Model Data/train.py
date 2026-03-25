import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import your custom classes from your other files
from dataset_loader import TornadoDataset, get_dataset_stats, DATA_DIR, TARGET_VARS
from model import TornadoPredictor

def main():
    # --- 1. SETUP HARDWARE ---
    # Automatically use the GPU if you have one, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 2. SETUP DATA ---
    TARGET_DATES_FILE = Path("Model Data/target_tornadic_dates.txt")
    
    # Initialize the raw dataset and get the normalization stats
    raw_dataset = TornadoDataset(DATA_DIR, TARGET_VARS, TARGET_DATES_FILE)
    mean, std = get_dataset_stats(raw_dataset)
    
    # Create the normalizer function
    def normalize_tensor(tensor):
        return (tensor - mean.squeeze(0)) / (std.squeeze(0) + 1e-7)
    
    # Create the final DataLoader (Conveyor Belt)
    ml_dataset = TornadoDataset(DATA_DIR, TARGET_VARS, TARGET_DATES_FILE, transform=normalize_tensor)
    
   # --- SCALED UP DATALOADER ---
    # batch_size=32: Processes 32 maps at a time (lower to 16 if your computer runs out of RAM)
    # num_workers=4: Uses 4 CPU cores to load files in the background
    # pin_memory=True: Speeds up the transfer of data from your RAM to your GPU
    dataloader = DataLoader(
        ml_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # --- 3. SETUP MODEL & LEARNING TOOLS ---
    model = TornadoPredictor(num_channels=len(TARGET_VARS)).to(device)
    
    # Binary Cross Entropy Loss: The math function that grades the model's accuracy (0 to 1)
    criterion = nn.BCELoss() 
    
    # The Optimizer (Adam): The mechanic that updates the model's weights based on the grade
    # lr = Learning Rate (how big of a step the model takes when adjusting its brain)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. THE TRAINING LOOP ---
    epochs = 10 # How many times the model reads the entire textbook
    
    print("\nStarting Training...\n" + "="*30)
    
    for epoch in range(epochs):
        model.train() # Tell the model it is in learning mode
        running_loss = 0.0
        
        for batch_idx, (maps, labels) in enumerate(dataloader):
            # Move the data to your CPU/GPU
            maps, labels = maps.to(device), labels.to(device)
            
            # Step 1: Zero the gradients (clear the model's memory from the last batch)
            optimizer.zero_grad()
            
            # Step 2: Forward Pass (Make a guess)
            predictions = model(maps)
            
            # Step 3: Calculate Loss (Grade the guess against the actual labels)
            loss = criterion(predictions, labels)
            
            # Step 4: Backward Pass (Calculate how much to change each brain connection)
            loss.backward()
            
            # Step 5: Optimizer Step (Actually apply the changes to the brain)
            optimizer.step()
            
            running_loss += loss.item()
            
        # Calculate the average loss for this epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    # --- 5. SAVE THE BRAIN ---
    torch.save(model.state_dict(), "Model Data/tornado_predictor_weights.pth")
    print("\nTraining Complete! Model saved as 'Model Data/tornado_predictor_weights.pth'")

if __name__ == "__main__":
    main()