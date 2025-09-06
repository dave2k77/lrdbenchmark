#!/usr/bin/env python3
"""
Simple test for the Production ML System.

This script tests the core functionality without complex data generation.
"""

import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_production_system():
    """Test the production ML system with simple data."""
    logger.info("🚀 Testing Production ML System")
    
    try:
        # Import the production system
        from lrdbenchmark.analysis.machine_learning.production_ml_system import (
            ProductionMLSystem, ProductionConfig
        )
        
        # Create a simple configuration
        config = ProductionConfig(
            model_type="cnn",
            input_length=100,
            hidden_dims=[32, 16],
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=16,
            epochs=5,  # Very short for testing
            early_stopping_patience=3,
            validation_split=0.2,
            use_jax=True,
            use_torch=True,
            use_numba=True,
            framework_priority=['torch', 'numba']  # Skip JAX for now
        )
        
        # Create the system
        system = ProductionMLSystem(config)
        logger.info(f"✅ Created production system with framework: {system.framework}")
        
        # Generate simple training data
        n_samples = 50
        seq_len = 100
        X = np.random.randn(n_samples, seq_len)
        y = np.random.uniform(0.2, 0.8, n_samples)
        
        logger.info(f"📊 Generated training data: {X.shape}, {y.shape}")
        
        # Train the model
        logger.info("🏋️ Training model...")
        start_time = time.time()
        result = system.train(X, y)
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"📈 Final MSE: {result['performance_metrics']['mse']:.4f}")
        
        # Test prediction
        logger.info("🔮 Testing prediction...")
        test_data = np.random.randn(seq_len)
        start_time = time.time()
        prediction = system.predict(test_data)
        prediction_time = time.time() - start_time
        
        logger.info(f"✅ Prediction completed in {prediction_time*1000:.2f}ms")
        logger.info(f"📊 Hurst estimate: {prediction.hurst_parameter:.3f}")
        logger.info(f"🎯 Framework used: {prediction.optimization_framework}")
        
        # Test batch prediction
        logger.info("📦 Testing batch prediction...")
        test_batch = np.random.randn(5, seq_len)
        start_time = time.time()
        batch_predictions = system.batch_predict([test_batch[i] for i in range(5)])
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch prediction completed in {batch_time*1000:.2f}ms")
        logger.info(f"📊 Batch size: {len(batch_predictions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_framework_availability():
    """Test which frameworks are available."""
    logger.info("🔍 Testing Framework Availability")
    
    frameworks = {}
    
    # Test JAX
    try:
        import jax
        import jax.numpy as jnp
        logger.info("  ✅ JAX available")
        frameworks['jax'] = True
    except ImportError:
        logger.info("  ❌ JAX not available")
        frameworks['jax'] = False
    
    # Test PyTorch
    try:
        import torch
        logger.info("  ✅ PyTorch available")
        frameworks['torch'] = True
    except ImportError:
        logger.info("  ❌ PyTorch not available")
        frameworks['torch'] = False
    
    # Test Numba
    try:
        import numba
        logger.info("  ✅ Numba available")
        frameworks['numba'] = True
    except ImportError:
        logger.info("  ❌ Numba not available")
        frameworks['numba'] = False
    
    return frameworks

def main():
    """Main test function."""
    logger.info("🧪 Starting Simple Production ML System Test")
    
    # Test framework availability
    frameworks = test_framework_availability()
    
    if not any(frameworks.values()):
        logger.error("❌ No ML frameworks available. Please install PyTorch or Numba.")
        return False
    
    # Test production system
    if not test_production_system():
        logger.error("❌ Production system test failed")
        return False
    
    logger.info("🎉 All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
