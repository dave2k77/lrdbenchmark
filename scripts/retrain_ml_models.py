import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import joblib
from pathlib import Path

from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise as FGNModel

def generate_training_data(n_samples=1000, n_series=100):
    """Generates training data for the ML estimators."""
    X, y = [], []
    for _ in range(n_series):
        h = np.random.uniform(0.1, 0.9)
        fgn = FGNModel(H=h)
        series = fgn.generate(length=n_samples)
        
        # Simple features: mean, std, and a few autocorrelation coefficients
        acf_vals = np.fft.ifft(np.abs(np.fft.fft(series))**2).real
        features = [
            np.mean(series),
            np.std(series),
            acf_vals[1],
            acf_vals[5],
            acf_vals[10],
        ]
        
        X.append(features)
        y.append(h)
    return np.array(X), np.array(y)

def train_and_save_models():
    """Trains and saves the scikit-learn based estimators."""
    print("Generating training data...")
    X_train, y_train = generate_training_data(n_series=500)

    models_to_train = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "svr": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    for name, model in models_to_train.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        
        # The filename should match what the estimators expect
        filename = output_dir / f"{name}_estimator.joblib"
        joblib.dump(model, filename)
        print(f"Saved trained {name} model to {filename}")

if __name__ == "__main__":
    train_and_save_models()
