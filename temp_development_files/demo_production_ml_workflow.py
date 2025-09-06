#!/usr/bin/env python3
"""
Comprehensive Demonstration of Production ML Workflow.

This script demonstrates the complete train-once, apply-many workflow with:
- JAX priority, PyTorch fallback, Numba optimization
- Production-ready model training and deployment
- Efficient inference with caching and batching
- Framework comparison and performance analysis
"""

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_training_data():
    """Generate realistic training data for LRD estimation."""
    logger.info("📊 Generating realistic training data...")
    
    # Generate data with known Hurst parameters
    n_samples = 200
    seq_len = 500
    X = []
    y = []
    
    # Generate fractional Brownian motion with different Hurst parameters
    hurst_values = np.linspace(0.2, 0.8, 10)
    
    for hurst in hurst_values:
        for _ in range(n_samples // len(hurst_values)):
            # Generate FBM-like data
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
    
    logger.info(f"✅ Generated {len(X)} samples with shape {X.shape}")
    return X, y

def test_framework_performance():
    """Test performance across different frameworks."""
    logger.info("🏃 Testing Framework Performance")
    
    from lrdbenchmark.analysis.machine_learning.production_ml_system import (
        ProductionMLSystem, ProductionConfig
    )
    
    # Generate test data
    X, y = generate_realistic_training_data()
    
    # Test different frameworks
    frameworks = ['jax', 'torch', 'numba']
    results = {}
    
    for framework in frameworks:
        logger.info(f"  Testing {framework.upper()} framework...")
        
        try:
            # Create configuration
            config = ProductionConfig(
                model_type="cnn",
                input_length=500,
                hidden_dims=[64, 32],
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=10,  # Short for demo
                early_stopping_patience=5,
                validation_split=0.2,
                use_jax=framework == 'jax',
                use_torch=framework == 'torch',
                use_numba=framework == 'numba',
                framework_priority=[framework]
            )
            
            # Create and train system
            system = ProductionMLSystem(config)
            
            # Train
            start_time = time.time()
            training_result = system.train(X, y)
            training_time = time.time() - start_time
            
            # Test inference speed
            test_data = X[0]  # Use first sample
            inference_times = []
            
            for _ in range(10):  # Test 10 times
                start_time = time.time()
                prediction = system.predict(test_data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            
            results[framework] = {
                'framework': framework,
                'training_time': training_time,
                'training_mse': training_result['performance_metrics']['mse'],
                'avg_inference_time': avg_inference_time,
                'inference_time_std': np.std(inference_times),
                'success': True,
                'prediction_sample': prediction.hurst_parameter
            }
            
            logger.info(f"    ✅ {framework.upper()}: Training={training_time:.2f}s, "
                       f"Inference={avg_inference_time*1000:.2f}ms, MSE={training_result['performance_metrics']['mse']:.4f}")
            
        except Exception as e:
            logger.warning(f"    ❌ {framework.upper()} failed: {e}")
            results[framework] = {
                'framework': framework,
                'success': False,
                'error': str(e)
            }
    
    return results

def demonstrate_train_once_apply_many():
    """Demonstrate the train-once, apply-many workflow."""
    logger.info("🎯 Demonstrating Train-Once, Apply-Many Workflow")
    
    from lrdbenchmark.analysis.machine_learning.production_ml_system import (
        ProductionMLSystem, ProductionConfig
    )
    
    # Step 1: Train the model once
    logger.info("📚 Step 1: Training model once...")
    
    X, y = generate_realistic_training_data()
    
    config = ProductionConfig(
        model_type="cnn",
        input_length=500,
        hidden_dims=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        early_stopping_patience=5,
        validation_split=0.2,
        framework_priority=['torch', 'numba']  # Skip JAX for stability
    )
    
    system = ProductionMLSystem(config)
    
    start_time = time.time()
    training_result = system.train(X, y)
    training_time = time.time() - start_time
    
    logger.info(f"✅ Model trained in {training_time:.2f} seconds")
    logger.info(f"📈 Final MSE: {training_result['performance_metrics']['mse']:.4f}")
    
    # Step 2: Apply many times (simulate production usage)
    logger.info("🚀 Step 2: Applying model many times...")
    
    # Generate test data
    n_test_samples = 100
    test_data = []
    true_hurst = []
    
    for _ in range(n_test_samples):
        hurst = np.random.uniform(0.2, 0.8)
        t = np.linspace(0, 1, 500)
        dt = t[1] - t[0]
        dB = np.random.normal(0, np.sqrt(dt), 500)
        fbm = np.cumsum(dB) * (dt ** hurst)
        test_data.append(fbm)
        true_hurst.append(hurst)
    
    # Single predictions
    logger.info("  Testing single predictions...")
    single_times = []
    predictions = []
    
    for i, data in enumerate(test_data[:10]):  # Test first 10
        start_time = time.time()
        result = system.predict(data)
        inference_time = time.time() - start_time
        
        single_times.append(inference_time)
        predictions.append(result.hurst_parameter)
        
        if i < 3:  # Show first 3 results
            logger.info(f"    Sample {i+1}: True={true_hurst[i]:.3f}, "
                       f"Predicted={result.hurst_parameter:.3f}, "
                       f"Time={inference_time*1000:.2f}ms")
    
    # Batch predictions
    logger.info("  Testing batch predictions...")
    start_time = time.time()
    batch_results = system.batch_predict(test_data[:20])  # Test first 20
    batch_time = time.time() - start_time
    
    batch_predictions = [r.hurst_parameter for r in batch_results]
    
    # Performance analysis
    single_avg_time = np.mean(single_times)
    batch_avg_time = batch_time / len(batch_results)
    
    logger.info(f"📊 Performance Summary:")
    logger.info(f"  Single prediction: {single_avg_time*1000:.2f}ms average")
    logger.info(f"  Batch prediction: {batch_avg_time*1000:.2f}ms average")
    logger.info(f"  Batch efficiency: {single_avg_time/batch_avg_time:.1f}x faster")
    
    # Accuracy analysis
    mae_single = np.mean(np.abs(np.array(predictions) - np.array(true_hurst[:10])))
    mae_batch = np.mean(np.abs(np.array(batch_predictions) - np.array(true_hurst[:20])))
    
    logger.info(f"📈 Accuracy Summary:")
    logger.info(f"  Single prediction MAE: {mae_single:.4f}")
    logger.info(f"  Batch prediction MAE: {mae_batch:.4f}")
    
    return {
        'training_time': training_time,
        'training_mse': training_result['performance_metrics']['mse'],
        'single_inference_time': single_avg_time,
        'batch_inference_time': batch_avg_time,
        'batch_efficiency': single_avg_time / batch_avg_time,
        'single_mae': mae_single,
        'batch_mae': mae_batch
    }

def demonstrate_caching():
    """Demonstrate prediction caching."""
    logger.info("💾 Demonstrating Prediction Caching")
    
    from lrdbenchmark.analysis.machine_learning.production_ml_system import (
        ProductionMLSystem, ProductionConfig
    )
    
    # Create system with caching enabled
    config = ProductionConfig(
        model_type="cnn",
        input_length=500,
        hidden_dims=[32, 16],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=16,
        epochs=5,
        early_stopping_patience=3,
        validation_split=0.2,
        cache_predictions=True,
        framework_priority=['torch']
    )
    
    system = ProductionMLSystem(config)
    
    # Generate training data
    X, y = generate_realistic_training_data()
    
    # Train model
    system.train(X, y)
    
    # Test caching
    test_data = X[0]
    
    # First prediction (cache miss)
    start_time = time.time()
    result1 = system.predict(test_data)
    time1 = time.time() - start_time
    
    # Second prediction (cache hit)
    start_time = time.time()
    result2 = system.predict(test_data)
    time2 = time.time() - start_time
    
    logger.info(f"📊 Caching Results:")
    logger.info(f"  First prediction (cache miss): {time1*1000:.2f}ms")
    logger.info(f"  Second prediction (cache hit): {time2*1000:.2f}ms")
    logger.info(f"  Cache speedup: {time1/time2:.1f}x faster")
    logger.info(f"  Cache hit: {result2.cache_hit}")
    
    return {
        'cache_miss_time': time1,
        'cache_hit_time': time2,
        'cache_speedup': time1 / time2
    }

def create_performance_visualization(results):
    """Create performance visualization."""
    logger.info("📊 Creating Performance Visualization")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        logger.warning("No successful results to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Production ML System Performance Comparison', fontsize=16)
    
    frameworks = list(successful_results.keys())
    
    # Training time comparison
    training_times = [successful_results[f]['training_time'] for f in frameworks]
    axes[0, 0].bar(frameworks, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('Training Time Comparison')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_yscale('log')
    
    # Inference time comparison
    inference_times = [successful_results[f]['avg_inference_time'] * 1000 for f in frameworks]
    axes[0, 1].bar(frameworks, inference_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title('Inference Time Comparison')
    axes[0, 1].set_ylabel('Time (milliseconds)')
    
    # Training MSE comparison
    training_mses = [successful_results[f]['training_mse'] for f in frameworks]
    axes[1, 0].bar(frameworks, training_mses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_title('Training MSE Comparison')
    axes[1, 0].set_ylabel('MSE')
    
    # Performance summary
    axes[1, 1].text(0.1, 0.8, 'Performance Summary:', fontsize=14, fontweight='bold')
    
    y_pos = 0.7
    for framework in frameworks:
        result = successful_results[framework]
        summary_text = f"{framework.upper()}:\n"
        summary_text += f"  Training: {result['training_time']:.2f}s\n"
        summary_text += f"  Inference: {result['avg_inference_time']*1000:.2f}ms\n"
        summary_text += f"  MSE: {result['training_mse']:.4f}\n"
        
        axes[1, 1].text(0.1, y_pos, summary_text, fontsize=10)
        y_pos -= 0.15
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "production_ml_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Performance visualization saved to {output_path}")

def main():
    """Main demonstration function."""
    logger.info("🎬 Starting Production ML Workflow Demonstration")
    
    # Create output directory
    Path("demo_output").mkdir(exist_ok=True)
    
    try:
        # Test framework performance
        logger.info("=" * 60)
        framework_results = test_framework_performance()
        
        # Demonstrate train-once, apply-many
        logger.info("=" * 60)
        workflow_results = demonstrate_train_once_apply_many()
        
        # Demonstrate caching
        logger.info("=" * 60)
        caching_results = demonstrate_caching()
        
        # Create visualization
        logger.info("=" * 60)
        create_performance_visualization(framework_results)
        
        # Save results
        results_summary = {
            'framework_performance': framework_results,
            'workflow_performance': workflow_results,
            'caching_performance': caching_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('demo_output/production_ml_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎉 Production ML Workflow Demonstration Completed!")
        logger.info("📁 Results saved to demo_output/")
        
        # Print summary
        logger.info("📊 Summary:")
        logger.info(f"  Frameworks tested: {len([k for k, v in framework_results.items() if v.get('success', False)])}")
        logger.info(f"  Training time: {workflow_results['training_time']:.2f}s")
        logger.info(f"  Inference speed: {workflow_results['single_inference_time']*1000:.2f}ms")
        logger.info(f"  Batch efficiency: {workflow_results['batch_efficiency']:.1f}x")
        logger.info(f"  Cache speedup: {caching_results['cache_speedup']:.1f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
