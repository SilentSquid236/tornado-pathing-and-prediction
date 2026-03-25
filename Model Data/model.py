import torch
import torch.nn as nn

class TornadoPredictor(nn.Module):
    def __init__(self, num_channels=2):
        super(TornadoPredictor, self).__init__()
        
        # --- PART 1: The Feature Extractor ---
        # Scans the weather maps for physical patterns (gradients, boundaries)
        self.features = nn.Sequential(
            # Layer 1: Looks at the 2 input channels (Wind & Temp)
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: Combines basic patterns into complex weather features
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # This is a pro-trick: It forces the varying spatial grid down to a strict 8x8 size.
        # This prevents the model from crashing if your NetCDF file sizes ever change slightly.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # --- PART 2: The Classifier (Decision Maker) ---
        # Takes the extracted features and outputs a tornado probability
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Drops 50% of connections randomly to prevent overfitting/memorization
            nn.Linear(128, 1),
            nn.Sigmoid() # Squashes the final number into a strict probability between 0.0 (No) and 1.0 (Yes)
        )

    def forward(self, x):
        # 1. Pass data through the convolutional layers
        x = self.features(x)
        # 2. Standardize the grid size
        x = self.adaptive_pool(x)
        # 3. Flatten the 3D maps into a 1D array of numbers
        x = torch.flatten(x, 1) 
        # 4. Pass the array through the final decision layers
        x = self.classifier(x)
        return x

""" --- QUICK TEST ---
if __name__ == "__main__":
    # Create a fake batch of data mimicking your DataLoader output
    fake_batch = torch.randn(2, 2, 144, 119) 
    
    # Initialize the model
    model = TornadoPredictor(num_channels=2)
    
    # Push the fake data through the model
    predictions = model(fake_batch)
    
    print("Model initialized successfully!")
    print(f"Input Shape:  {fake_batch.shape}")
    print(f"Output Shape: {predictions.shape}")
    print(f"Predictions:  {predictions.detach().squeeze().tolist()}")"""