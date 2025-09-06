#!/usr/bin/env python3
"""
Test script for proper ML estimators (SVR, Gradient Boosting, Random Forest).

This script tests the newly implemented ML estimators to ensure they work correctly.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new ML estimators
from lrdbenchmark.analysis.machine_learning.svr_estimator import SVREstimator
from lrdbenchmark.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
from lrdbenchmark.analysis.machine_learning.random_forest_estimator import RandomForestEstimator

def generate_test_data(n_samples: int = 100, seq_len: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data with known Hurst parameters."""
    logger.info(f"Generating {n_samples} test samples of length {seq_len}")
    
    X = []
    y = []
    
    # Generate data with different Hurst parameters
    hurst_values = np.linspace(0.2, 0.8, 10)
    
    for hurst in hurst_values:
        for _ in range(n_samples // len(hurst_values)):
            # Generate fractional Brownian motion-like data
            t = np.linspace(0, 1, seq_len)
            dt = t[1] - t[0]
            
            # Generate increments with Hurst scaling
            dB = np.random.normal(0, np.sqrt(dt), seq_len)
            fbm = np.cumsum(dB) * (dt ** hurst)
            
            # Add some noise
            noise = np.random.normal(0, 0.1, seq_len)
            data = fbm + noise
            
            X.append(data)
            y.append(hurst)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Generated {len(X)} samples with shape {X.shape}")
    return X, y

def test_estimator(estimator, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Test a single estimator."""
    logger.info(f"Testing {name} estimator...")
    
    results = {
        'name': name,
        'training_success': False,
        'prediction_success': False,
        'training_time': 0.0,
        'prediction_time': 0.0,
        'training_results': None,
        'predictions': [],
        'errors': []
    }
    
    try:
        # Test training
        logger.info(f"  Training {name}...")
        start_time = time.time()
        training_results = estimator.train(X, y, validation_split=0.2)
        training_time = time.time() - start_time
        
        results['training_success'] = True
        results['training_time'] = training_time
        results['training_results'] = training_results
        
        logger.info(f"    ✅ {name} training completed in {training_time:.2f}s")
        logger.info(f"    Training MSE: {training_results.get('mse', 0):.4f}")
        logger.info(f"    Training R²: {training_results.get('r2', 0):.4f}")
        
        # Test predictions
        logger.info(f"  Testing {name} predictions...")
        start_time = time.time()
        
        predictions = []
        errors = []
        
        for i in range(min(10, len(X))):  # Test on first 10 samples
            try:
                prediction = estimator.predict(X[i])
                predictions.append(prediction)
                error = abs(prediction - y[i])
                errors.append(error)
            except Exception as e:
                logger.error(f"    Prediction failed for sample {i}: {e}")
                errors.append(float('inf'))
        
        prediction_time = time.time() - start_time
        
        results['prediction_success'] = True
        results['prediction_time'] = prediction_time
        results['predictions'] = predictions
        results['errors'] = errors
        
        if errors:
            mean_error = np.mean([e for e in errors if e != float('inf')])
            logger.info(f"    ✅ {name} predictions completed in {prediction_time:.4f}s")
            logger.info(f"    Mean absolute error: {mean_error:.4f}")
        
        # Test estimation method
        logger.info(f"  Testing {name} estimation method...")
        try:
            estimation_result = estimator.estimate(X[0])
            logger.info(f"    ✅ {name} estimation method works")
            logger.info(f"    Estimated Hurst: {estimation_result.get('hurst_parameter', 0):.4f}")
            logger.info(f"    Method: {estimation_result.get('method', 'unknown')}")
        except Exception as e:
            logger.error(f"    ❌ {name} estimation method failed: {e}")
        
        # Test model info
        try:
            model_info = estimator.get_model_info()
            logger.info(f"    Model info: {model_info.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"    ❌ {name} model info failed: {e}")
        
    except Exception as e:
        logger.error(f"    ❌ {name} testing failed: {e}")
        results['error'] = str(e)
    
    return results

def main():
    """Main test function."""
    logger.info("🚀 Testing Proper ML Estimators")
    
    try:
        # Generate test data
        logger.info("=" * 60)
        X, y = generate_test_data(n_samples=50, seq_len=500)
        
        # Initialize estimators
        estimators = {
            'SVR': SVREstimator(kernel='rbf', C=1.0, gamma='scale'),
            'GradientBoosting': GradientBoostingEstimator(n_estimators=50, learning_rate=0.1),
            'RandomForest': RandomForestEstimator(n_estimators=50, max_depth=5)
        }
        
        # Test each estimator
        results = {}
        for name, estimator in estimators.items():
            logger.info("=" * 60)
            results[name] = test_estimator(estimator, name, X, y)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 Test Results Summary:")
        
        for name, result in results.items():
            logger.info(f"  {name}:")
            logger.info(f"    Training: {'✅' if result['training_success'] else '❌'}")
            logger.info(f"    Prediction: {'✅' if result['prediction_success'] else '❌'}")
            if result['training_success']:
                logger.info(f"    Training time: {result['training_time']:.2f}s")
                if result['training_results']:
                    logger.info(f"    Training MSE: {result['training_results'].get('mse', 0):.4f}")
                    logger.info(f"    Training R²: {result['training_results'].get('r2', 0):.4f}")
            if result['prediction_success'] and result['errors']:
                valid_errors = [e for e in result['errors'] if e != float('inf')]
                if valid_errors:
                    logger.info(f"    Mean prediction error: {np.mean(valid_errors):.4f}")
        
        # Find best estimator
        best_estimator = None
        best_mse = float('inf')
        
        for name, result in results.items():
            if result['training_success'] and result['training_results']:
                mse = result['training_results'].get('mse', float('inf'))
                if mse < best_mse:
                    best_mse = mse
                    best_estimator = name
        
        if best_estimator:
            logger.info(f"🏆 Best estimator: {best_estimator} (MSE: {best_mse:.4f})")
        
        logger.info("🎉 ML Estimators Testing Completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
