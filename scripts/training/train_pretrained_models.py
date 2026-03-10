import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from lrdbenchmark.generation.time_series_generator import TimeSeriesGenerator
from lrdbenchmark.models.pretrained_models.cnn_pretrained import SimpleCNN1D
from lrdbenchmark.models.pretrained_models.transformer_pretrained import SimpleTransformer
from lrdbenchmark.models.pretrained_models.lstm_pretrained import LSTMModel

def extract_ml_features(data):
    """Extract traditional features for ML models: variance ratio, spectral slope, autocorrelation."""
    features = []
    for i in range(data.shape[0]):
        series = data[i].copy()
        
        # 1. Variance ratio across segments
        segment_size = max(10, len(series) // 4)
        segments = [
            series[j : j + segment_size]
            for j in range(0, len(series), segment_size)
            if j + segment_size <= len(series)
        ]
        
        var_ratio = 1.0
        if segments:
            variances = [np.var(seg) for seg in segments]
            var_ratio = np.std(variances) / (np.mean(variances) + 1e-8)
            
        # 2. Spectral slope
        fft_vals = np.abs(np.fft.fft(series))
        freqs = np.fft.fftfreq(len(series))
        positive_freqs = freqs > 0
        spectral_slope = -1.0
        if np.sum(positive_freqs) > 1:
            log_freqs = np.log(freqs[positive_freqs] + 1e-8)
            log_fft = np.log(fft_vals[positive_freqs] + 1e-8)
            if len(log_freqs) > 1:
                spectral_slope = np.polyfit(log_freqs, log_fft, 1)[0]
                
        # 3. Autocorrelation
        autocorr = 0.0
        if len(series) > 1:
            cc = np.corrcoef(series[:-1], series[1:])[0, 1]
            if not np.isnan(cc):
                autocorr = cc
                
        features.append([var_ratio, spectral_slope, autocorr])
        
    return np.array(features)

def generate_dataset(num_samples=4000, length=1024, seed=42):
    print(f"Generating {num_samples} fGn series of length {length}...")
    generator = TimeSeriesGenerator(random_state=seed)
    
    X = []
    y = []
    
    rng = np.random.default_rng(seed)
    
    # Stratified-ish generation of H
    h_values = rng.uniform(0.05, 0.95, size=num_samples)
    
    for h in tqdm(h_values, desc="Generating fGns"):
        res = generator.generate(model='fgn', length=length, params={'H': h}, preprocess=True)
        # normalize
        signal = res['signal']
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        X.append(signal)
        y.append(h)
        
    return np.array(X), np.array(y)

def train_ml_models(X_train, y_train, model_dir):
    print("-- Extracting ML features --")
    features_train = extract_ml_features(X_train)
    
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(features_train, y_train)
    joblib.dump(rf, os.path.join(model_dir, 'rf_pretrained.joblib'))
    
    print("Training SVR...")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    svr.fit(features_train, y_train)
    joblib.dump(svr, os.path.join(model_dir, 'svr_pretrained.joblib'))
    
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(features_train, y_train)
    joblib.dump(gb, os.path.join(model_dir, 'gb_pretrained.joblib'))
    
def train_nn_model(model, X_train, y_train, epochs=20, batch_size=32, device='cpu', filename='model.pth', unsqueeze_dim=None):
    print(f"Training {filename}...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    
    if unsqueeze_dim is not None:
        X_tensor = X_tensor.unsqueeze(unsqueeze_dim)
        
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
    # Remove the .pth from the path because we received the full name
    torch.save(model.state_dict(), filename)
    print(f"Saved {filename}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lrdbenchmark', 'assets', 'models'))
    os.makedirs(model_dir, exist_ok=True)
    
    X, y = generate_dataset(num_samples=4000, length=1024, seed=42)
    
    # ML Models (takes full 1024 features to extract from)
    train_ml_models(X, y, model_dir)
    
    # CNN (Length 500)
    cnn_model = SimpleCNN1D(input_length=500).to(device)
    train_nn_model(
        cnn_model, 
        X[:, :500], y, 
        epochs=15, 
        device=device, 
        filename=os.path.join(model_dir, 'cnn_pretrained.pth'),
        unsqueeze_dim=1 # CNN wants (batch, 1, seq_len)
    )
    
    # Transformer (Length 500)
    transformer_model = SimpleTransformer(input_length=500).to(device)
    train_nn_model(
        transformer_model, 
        X[:, :500], y, 
        epochs=15, 
        device=device, 
        filename=os.path.join(model_dir, 'transformer_pretrained.pth'),
        unsqueeze_dim=-1 # Transformer wants (batch, seq_len, 1) or handled inside model? Let's check Transformer later, wait it's (batch, seq_len, 1) inside transformer view?
    )
    
    # LSTM (Length 1024)
    lstm_model = LSTMModel(input_size=1).to(device)
    train_nn_model(
        lstm_model, 
        X, y, 
        epochs=15, 
        device=device, 
        filename=os.path.join(model_dir, 'lstm_model.pth'),
        unsqueeze_dim=-1 # LSTM wants (batch, seq_len, 1)
    )
    
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
