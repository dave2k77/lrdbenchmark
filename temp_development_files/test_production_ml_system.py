#!/usr/bin/env python3
"""
Test script for the Production ML System with Train-Once, Apply-Many workflow.

This script demonstrates the complete pipeline from training to deployment.
"""

import numpy as np
import time
import logging
from pathlib import Path

# Import our production system
from lrdbenchmark.analysis.machine_learning.train_once_apply_many import (
    TrainOnceApplyManyPipeline,
    TrainingDataConfig,
    ModelTrainingConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_production_ml_system():
    """Test the complete production ML system."""
    logger.info("🚀 Starting Production ML System Test")
    
    # Configuration for training data
    training_data_config = TrainingDataConfig(
        n_samples_per_model=100,  # Reduced for testing
        sequence_lengths=[100, 250, 500],
        hurst_range=(0.2, 0.8),
        noise_levels=[0.0, 0.01, 0.05],
        contamination_scenarios=['pure', 'gaussian_noise', 'outliers']
    )
    
    # Configuration for model training
    model_training_config = ModelTrainingConfig(
        model_types=['cnn', 'transformer'],
        input_length=500,
        hidden_dims=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=16,  # Smaller for testing
        epochs=20,  # Reduced for testing
        early_stopping_patience=5,
        validation_split=0.2,
        prefer_jax=True,
        prefer_torch=True,
        prefer_numba=True
    )
    
    # Create pipeline
    pipeline = TrainOnceApplyManyPipeline(
        training_data_config=training_data_config,
        model_training_config=model_training_config,
        registry_path="test_models/registry.json"
    )
    
    # Step 1: Run training pipeline
    logger.info("📚 Step 1: Running Training Pipeline")
    start_time = time.time()
    
    try:
        training_results = pipeline.run_training_pipeline()
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"📊 Trained {len(training_results)} models")
        
        # Print training results
        for result in training_results:
            logger.info(f"  - {result.model_type} ({result.framework}): "
                       f"MSE={result.performance_metrics['mse']:.4f}, "
                       f"Training time={result.training_time:.2f}s")
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False
    
    # Step 2: Deploy best model
    logger.info("🚀 Step 2: Deploying Best Model")
    
    try:
        deployed_model = pipeline.deploy_best_model()
        if deployed_model:
            logger.info("✅ Model deployed successfully")
        else:
            logger.error("❌ Failed to deploy model")
            return False
    
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False
    
    # Step 3: Test predictions
    logger.info("🔮 Step 3: Testing Predictions")
    
    try:
        # Generate test data
        test_data = generate_test_data()
        
        # Single prediction
        logger.info("  Testing single prediction...")
        start_time = time.time()
        result = pipeline.predict(test_data[0])
        prediction_time = time.time() - start_time
        
        if result:
            logger.info(f"  ✅ Single prediction: Hurst={result.hurst_parameter:.3f}, "
                       f"Time={prediction_time*1000:.2f}ms, Framework={result.optimization_framework}")
        else:
            logger.error("  ❌ Single prediction failed")
            return False
        
        # Batch prediction
        logger.info("  Testing batch prediction...")
        start_time = time.time()
        batch_results = pipeline.batch_predict(test_data[:5])
        batch_time = time.time() - start_time
        
        if batch_results:
            logger.info(f"  ✅ Batch prediction: {len(batch_results)} predictions, "
                       f"Time={batch_time*1000:.2f}ms, "
                       f"Avg per prediction={batch_time*1000/len(batch_results):.2f}ms")
        else:
            logger.error("  ❌ Batch prediction failed")
            return False
    
    except Exception as e:
        logger.error(f"❌ Prediction testing failed: {e}")
        return False
    
    # Step 4: Performance summary
    logger.info("📈 Step 4: Performance Summary")
    
    try:
        summary = pipeline.get_model_performance_summary()
        
        logger.info(f"📊 Total models trained: {summary['total_models']}")
        logger.info(f"📊 Model types: {summary['model_types']}")
        logger.info(f"📊 Frameworks used: {summary['frameworks']}")
        
        logger.info("🏆 Best performers:")
        for model_type, info in summary['best_performers'].items():
            logger.info(f"  - {model_type}: {info['framework']} (MSE={info['mse']:.4f})")
    
    except Exception as e:
        logger.error(f"❌ Performance summary failed: {e}")
        return False
    
    logger.info("🎉 Production ML System Test Completed Successfully!")
    return True

def generate_test_data():
    """Generate test data for prediction testing."""
    logger.info("  Generating test data...")
    
    # Generate synthetic time series with known Hurst parameters
    test_data = []
    hurst_values = [0.3, 0.5, 0.7]
    
    for hurst in hurst_values:
        # Generate fractional Brownian motion
        n = 500
        t = np.linspace(0, 1, n)
        
        # Simple FBM generation
        dt = t[1] - t[0]
        dB = np.random.normal(0, np.sqrt(dt), n)
        
        # Apply Hurst scaling
        fbm = np.cumsum(dB)
        fbm = fbm * (dt ** hurst)
        
        test_data.append(fbm)
    
    return test_data

def test_framework_availability():
    """Test which frameworks are available."""
    logger.info("🔍 Testing Framework Availability")
    
    # Test JAX
    try:
        import jax
        import jax.numpy as jnp
        logger.info("  ✅ JAX available")
        jax_available = True
    except ImportError:
        logger.info("  ❌ JAX not available")
        jax_available = False
    
    # Test PyTorch
    try:
        import torch
        logger.info("  ✅ PyTorch available")
        torch_available = True
    except ImportError:
        logger.info("  ❌ PyTorch not available")
        torch_available = False
    
    # Test Numba
    try:
        import numba
        logger.info("  ✅ Numba available")
        numba_available = True
    except ImportError:
        logger.info("  ❌ Numba not available")
        numba_available = False
    
    return {
        'jax': jax_available,
        'torch': torch_available,
        'numba': numba_available
    }

def test_individual_components():
    """Test individual components of the system."""
    logger.info("🧪 Testing Individual Components")
    
    # Test data generation
    logger.info("  Testing data generation...")
    try:
        from lrdbenchmark.analysis.machine_learning.train_once_apply_many import TrainingDataGenerator, TrainingDataConfig
        
        config = TrainingDataConfig(n_samples_per_model=10, sequence_lengths=[100])
        generator = TrainingDataGenerator(config)
        X, y, metadata = generator.generate_training_data()
        
        logger.info(f"    ✅ Generated {len(X)} samples")
        logger.info(f"    ✅ Data shape: {X.shape}, Labels shape: {y.shape}")
        
    except Exception as e:
        logger.error(f"    ❌ Data generation failed: {e}")
        return False
    
    # Test model registry
    logger.info("  Testing model registry...")
    try:
        from lrdbenchmark.analysis.machine_learning.train_once_apply_many import ModelRegistry
        
        registry = ModelRegistry("test_models/test_registry.json")
        logger.info("    ✅ Model registry created")
        
    except Exception as e:
        logger.error(f"    ❌ Model registry failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    logger.info("🧪 Starting Production ML System Tests")
    
    # Test framework availability
    frameworks = test_framework_availability()
    
    if not any(frameworks.values()):
        logger.error("❌ No ML frameworks available. Please install JAX, PyTorch, or Numba.")
        return False
    
    # Test individual components
    if not test_individual_components():
        logger.error("❌ Individual component tests failed")
        return False
    
    # Test complete system
    if not test_production_ml_system():
        logger.error("❌ Production ML system test failed")
        return False
    
    logger.info("🎉 All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
