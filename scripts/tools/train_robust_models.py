
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
# Attempt to import ARFIMA, but fallback if fails easily
try:
    from lrdbenchmark.models.data_models.arfima.arfima_model import ARFIMA
    ARFIMA_AVAILABLE = True
except ImportError:
    ARFIMA_AVAILABLE = False
    print("ARFIMA model not found, training on FBM/FGN only.")

from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMModel
from lrdbenchmark.models.pretrained_models.gru_pretrained import GRUModel

def generate_robust_data(n_samples=20000, length=1024):
    """Generate large-scale, diverse training data."""
    print(f"Generating {n_samples} training samples (Target: Robust Models)...")
    X = []
    y = []
    
    rng = np.random.default_rng(42)
    
    for i in range(n_samples):
        # 50% chance of FBM/FGN, 50% ARFIMA (if available)
        use_arfima = ARFIMA_AVAILABLE and (rng.random() > 0.5)
        
        target_H = rng.uniform(0.1, 0.9)
        
        if use_arfima:
            # ARFIMA(0, d, 0)
            # H = d + 0.5  => d = H - 0.5
            d = target_H - 0.5
            # Ensure d is within stable range, usually (-0.5, 0.5)
            # ARFIMA generic model usually takes parameter dict
            try:
                model = ARFIMA(d=d) 
                data = model.generate(length=length, rng=rng)
            except Exception:
                # Fallback to fbm if generation fails
                fbm = FractionalBrownianMotion(H=target_H)
                data = fbm.generate(length=length+1, rng=rng)
                data = np.diff(data)
        else:
            # FBM (differenced to get fGn)
            fbm = FractionalBrownianMotion(H=target_H)
            data = fbm.generate(length=length+1, rng=rng)
            data = np.diff(data)
            
        # Robust Preprocessing
        # 1. Remove NaNs/Infs
        if not np.all(np.isfinite(data)):
             # Skip bad data
             continue
             
        # 2. Z-score normalize
        std_val = np.std(data)
        if std_val > 1e-8:
            data = (data - np.mean(data)) / std_val
        else:
             # Skip constant data
             continue
             
        X.append(data)
        y.append(target_H)
        
        if (i+1) % 1000 == 0:
            print(f"Generated {i+1}/{n_samples} samples")
        
    return np.array(X), np.array(y)

def train_network(model_class, model_name, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    print(f"\nTraining {model_name} (Robust Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model_class(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Datasets
    train_x = torch.Tensor(X_train).unsqueeze(-1).to(device)
    train_y = torch.Tensor(y_train).float().unsqueeze(-1).to(device)
    val_x = torch.Tensor(X_val).unsqueeze(-1).to(device)
    val_y = torch.Tensor(y_val).float().unsqueeze(-1).to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.join("lrdbenchmark", "assets", "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name.lower()}_model.pth")
            torch.save(model.state_dict(), save_path)
            
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to lrdbenchmark/assets/models/{model_name.lower()}_model.pth")

def main():
    # 1. Generate Data (20k samples)
    X, y = generate_robust_data(n_samples=20000, length=1024)
    
    # Shuffle and Split
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"Training Set: {len(X_train)}, Validation Set: {len(X_val)}")
    
    # 2. Train LSTM
    train_network(LSTMModel, "lstm", X_train, y_train, X_val, y_val, epochs=10)
    
    # 3. Train GRU
    train_network(GRUModel, "gru", X_train, y_train, X_val, y_val, epochs=10)

if __name__ == "__main__":
    main()
