
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMModel
from lrdbenchmark.models.pretrained_models.gru_pretrained import GRUModel

def generate_training_data(n_samples=1000, length=1024):
    """Generate synthetic training data."""
    print(f"Generating {n_samples} training samples...")
    X = []
    y = []
    
    fbm = FractionalBrownianMotion()
    rng = np.random.default_rng(42)
    
    for _ in range(n_samples):
        H = rng.uniform(0.1, 0.9)
        # Randomly choose between fBm (stationary increments) and fGn (stationary)
        # Ideally we train on differenced fBm (fGn) or standardize
        # For simplicity, let's generate fGn directly or difference fBm
        
        # Generating fGn indirectly via diff(fBm) is often stable
        data = fbm.generate(length=length+1, H=H, rng=rng)
        # Differencing to make it stationary (fGn-like)
        data = np.diff(data)
        
        # Z-score normalize
        data = (data - np.mean(data)) / np.std(data)
        
        X.append(data)
        y.append(H)
        
    return np.array(X), np.array(y)

def train_model(model_class, model_name, X_train, y_train, epochs=5):
    print(f"\nTraining {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model_class(input_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare data loader
    tensor_x = torch.Tensor(X_train).unsqueeze(-1).to(device) # (N, L, 1)
    tensor_y = torch.Tensor(y_train).float().unsqueeze(-1).to(device)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    # Save model
    save_dir = os.path.join("lrdbenchmark", "assets", "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.lower()}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved {model_name} to {save_path}")

def main():
    # 1. Generate Data
    X, y = generate_training_data(n_samples=500, length=1024)
    
    # 2. Train LSTM
    train_model(LSTMModel, "lstm", X, y, epochs=5)
    
    # 3. Train GRU
    train_model(GRUModel, "gru", X, y, epochs=5)

if __name__ == "__main__":
    main()
