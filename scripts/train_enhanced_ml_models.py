"""
Enhanced ML Model Training with Full Feature Extraction

This script trains ML estimators using the complete 76-feature extraction pipeline
which includes autocorrelation, spectral, wavelet, DFA, and entropy features.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.analysis.machine_learning.unified_feature_extractor import UnifiedFeatureExtractor


def generate_enhanced_training_data(n_samples_per_h=200, series_length=1024):
    """
    Generate diverse training data with full feature extraction.
    
    Parameters
    ----------
    n_samples_per_h : int
        Number of samples per Hurst value
    series_length : int
        Length of each time series
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, 76)
    y : np.ndarray
        Target Hurst values
    """
    print(f"Generating enhanced training data...")
    print(f"  - {n_samples_per_h} samples per Hurst value")
    print(f"  - Series length: {series_length}")
    
    X, y = [], []
    
    # Hurst values to sample
    hurst_values = np.linspace(0.1, 0.9, 17)  # 17 values from 0.1 to 0.9
    
    for h_idx, h in enumerate(hurst_values):
        print(f"  Generating H={h:.2f} ({h_idx+1}/{len(hurst_values)})...")
        
        for i in range(n_samples_per_h):
            # Randomly choose between FBM and FGN
            use_fbm = np.random.random() > 0.5
            
            try:
                if use_fbm:
                    model = FractionalBrownianMotion(H=h)
                    series = model.generate(length=series_length + 1)
                    # Convert FBM to fGn (increments)
                    series = np.diff(series)
                else:
                    model = FractionalGaussianNoise(H=h)
                    series = model.generate(length=series_length)
                
                # Add some random variations to increase diversity
                if np.random.random() > 0.7:
                    # Add small Gaussian noise
                    series = series + 0.05 * np.std(series) * np.random.randn(len(series))
                
                if np.random.random() > 0.8:
                    # Add weak trend
                    trend = 0.001 * np.arange(len(series)) * np.random.randn()
                    series = series + trend
                
                # Extract 76 features
                features = UnifiedFeatureExtractor.extract_features_76(series)
                
                # Skip if features contain NaN or Inf
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue
                    
                X.append(features)
                y.append(h)
                
            except Exception as e:
                print(f"    Warning: Failed to generate sample: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  Generated {len(X)} valid samples with {X.shape[1]} features each")
    return X, y


def train_enhanced_models(X_train, y_train, output_dir):
    """
    Train ML models with enhanced features and proper scaling.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    output_dir : Path
        Output directory for saved models
    """
    # Fit scaler
    print("\nFitting feature scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Handle any remaining NaN/Inf after scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Define models with optimized hyperparameters
    models_to_train = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "svr": SVR(
            kernel='rbf',
            C=10.0,
            epsilon=0.01,
            gamma='scale'
        ),
    }
    
    output_dir.mkdir(exist_ok=True, parents=True)
    feature_names = UnifiedFeatureExtractor.get_feature_names_76()
    
    for name, model in models_to_train.items():
        print(f"\nTraining {name} model...")
        model.fit(X_scaled, y_train)
        
        # Evaluate on training data
        y_pred = model.predict(X_scaled)
        mae = np.mean(np.abs(y_pred - y_train))
        print(f"  Training MAE: {mae:.4f}")
        
        # Save model with scaler and metadata
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': 76,
            'training_samples': len(X_train),
            'training_mae': mae
        }
        
        filename = output_dir / f"{name}_estimator.joblib"
        joblib.dump(model_package, filename)
        print(f"  Saved to {filename}")
        
        # Also save _fixed version for asset compatibility
        filename_fixed = output_dir / f"{name}_estimator_fixed.joblib"
        joblib.dump(model_package, filename_fixed)
        print(f"  Saved to {filename_fixed}")


def main():
    print("=" * 60)
    print("Enhanced ML Model Training with 76-Feature Extraction")
    print("=" * 60)
    
    # Generate training data
    X_train, y_train = generate_enhanced_training_data(
        n_samples_per_h=300,  # 300 samples per Hurst value
        series_length=1024
    )
    
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Train models
    output_dir = Path("models")
    train_enhanced_models(X_train, y_train, output_dir)
    
    # Copy to assets
    assets_dir = Path("lrdbenchmark/assets/models")
    assets_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCopying models to assets directory...")
    import shutil
    for f in output_dir.glob("*.joblib"):
        shutil.copy(f, assets_dir / f.name)
        print(f"  Copied {f.name}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
