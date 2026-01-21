"""
Enhanced Neural Network Training with Contamination Awareness

This script trains LSTM and GRU models on data that includes diverse contamination
patterns, enabling the models to learn contamination-robust Hurst estimation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.append(os.getcwd())

from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMModel
from lrdbenchmark.models.pretrained_models.gru_pretrained import GRUModel


def apply_random_contamination(data, rng):
    """Apply random contamination to training data for robustness."""
    contaminated = data.copy()
    n = len(data)
    
    # Randomly choose contamination type
    contam_type = rng.choice([
        'none', 'spikes', 'trend', 'noise', 'level_shift', 'missing'
    ], p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1])
    
    if contam_type == 'spikes':
        # Add spikes
        n_spikes = rng.integers(5, 30)
        spike_idx = rng.choice(n, size=min(n_spikes, n//10), replace=False)
        amplitude = rng.uniform(2.0, 5.0) * np.std(data)
        contaminated[spike_idx] += amplitude * rng.normal(0, 1, len(spike_idx))
        
    elif contam_type == 'trend':
        # Add trend
        trend_type = rng.choice(['linear', 'polynomial'])
        if trend_type == 'linear':
            slope = rng.uniform(0.001, 0.02)
            contaminated += slope * np.arange(n)
        else:
            x = np.arange(n) / n
            degree = rng.integers(2, 4)
            contaminated += 0.1 * np.std(data) * (x ** degree)
            
    elif contam_type == 'noise':
        # Add additional noise
        noise_std = rng.uniform(0.1, 0.5) * np.std(data)
        contaminated += noise_std * rng.normal(0, 1, n)
        
    elif contam_type == 'level_shift':
        # Add level shifts
        n_shifts = rng.integers(1, 5)
        shift_idx = sorted(rng.choice(n, size=n_shifts, replace=False))
        current_level = 0
        for i, idx in enumerate(shift_idx):
            shift = rng.uniform(0.5, 2.0) * np.std(data) * rng.choice([-1, 1])
            current_level += shift
            contaminated[idx:] += shift
            
    elif contam_type == 'missing':
        # Simulate missing data with interpolation
        n_missing = int(rng.uniform(0.05, 0.15) * n)
        missing_idx = rng.choice(n, size=n_missing, replace=False)
        # Replace with linear interpolation
        for idx in missing_idx:
            if idx == 0:
                contaminated[idx] = contaminated[1]
            elif idx == n - 1:
                contaminated[idx] = contaminated[n - 2]
            else:
                contaminated[idx] = (contaminated[idx - 1] + contaminated[idx + 1]) / 2
    
    return contaminated


def generate_enhanced_data(n_samples=25000, length=1024):
    """Generate diverse training data with contamination patterns."""
    print(f"Generating {n_samples} enhanced training samples...")
    X, y = [], []
    
    rng = np.random.default_rng(42)
    
    for i in range(n_samples):
        # Sample Hurst parameter
        target_H = rng.uniform(0.1, 0.9)
        
        # Randomly choose data type
        data_type = rng.choice(['fbm', 'fgn'], p=[0.6, 0.4])
        
        try:
            if data_type == 'fbm':
                fbm = FractionalBrownianMotion(H=target_H)
                data = fbm.generate(length=length + 1, rng=rng)
                data = np.diff(data)  # Convert to fGn
            else:
                fgn = FractionalGaussianNoise(H=target_H)
                data = fgn.generate(length=length, rng=rng)
            
            # Apply random contamination (60% of samples)
            if rng.random() > 0.4:
                data = apply_random_contamination(data, rng)
            
            # Handle any NaN/Inf
            if not np.all(np.isfinite(data)):
                continue
                
            # Z-score normalize
            std_val = np.std(data)
            if std_val > 1e-8:
                data = (data - np.mean(data)) / std_val
            else:
                continue
                
            X.append(data)
            y.append(target_H)
            
        except Exception as e:
            continue
        
        if (i + 1) % 2000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")
    
    print(f"  Total valid samples: {len(X)}")
    return np.array(X), np.array(y)


def train_network(model_class, model_name, X_train, y_train, X_val, y_val, 
                  epochs=15, batch_size=64, learning_rate=0.001):
    """Train neural network with enhanced settings."""
    print(f"\nTraining {model_name} (Enhanced Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model = model_class(input_size=1, hidden_size=128, num_layers=3, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Create datasets
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            
            # Calculate MAE for better understanding
            val_mae = torch.mean(torch.abs(val_outputs - val_y)).item()
        
        scheduler.step(val_loss)
        
        print(f"  Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = os.path.join("lrdbenchmark", "assets", "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name.lower()}_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    Saved best model (Val Loss: {val_loss:.6f})")
    
    print(f"  Training Complete. Best Validation Loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    print("=" * 60)
    print("Enhanced Neural Network Training")
    print("With Contamination-Aware Data Augmentation")
    print("=" * 60)
    
    # Generate enhanced data
    X, y = generate_enhanced_data(n_samples=25000, length=1024)
    
    # Shuffle and split
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.85)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"\nTraining Set: {len(X_train)}, Validation Set: {len(X_val)}")
    
    # Train LSTM
    train_network(LSTMModel, "lstm", X_train, y_train, X_val, y_val, epochs=15)
    
    # Train GRU
    train_network(GRUModel, "gru", X_train, y_train, X_val, y_val, epochs=15)
    
    print("\n" + "=" * 60)
    print("Enhanced Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
