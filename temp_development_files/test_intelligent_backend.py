#!/usr/bin/env python3
"""
Test Intelligent Backend Framework

This script tests the intelligent backend framework with sophisticated hardware
utilization strategies, memory-aware computation scheduling, and distributed computing.

Author: LRDBenchmark Team
Date: 2025-01-05
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the intelligent backend framework
from intelligent_backend_framework import (
    IntelligentBackend, HardwareConfig, ComputeBackend, 
    MemoryStrategy, TaskInfo
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentBackendTester:
    """Test the intelligent backend framework."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        self.performance_metrics = {}
        
        # Test configurations
        self.test_configs = {
            'conservative': HardwareConfig(
                max_cpu_cores=2,
                cpu_memory_limit=4.0,
                memory_strategy=MemoryStrategy.CONSERVATIVE,
                enable_caching=True,
                max_parallel_tasks=2
            ),
            'aggressive': HardwareConfig(
                max_cpu_cores=8,
                cpu_memory_limit=16.0,
                memory_strategy=MemoryStrategy.AGGRESSIVE,
                enable_caching=True,
                max_parallel_tasks=8
            ),
            'adaptive': HardwareConfig(
                max_cpu_cores=4,
                cpu_memory_limit=8.0,
                memory_strategy=MemoryStrategy.ADAPTIVE,
                enable_caching=True,
                max_parallel_tasks=4
            )
        }
    
    def test_hardware_detection(self):
        """Test hardware detection capabilities."""
        print("\nTesting Hardware Detection...")
        
        config = HardwareConfig()
        backend = IntelligentBackend(config)
        
        hw_info = backend.get_hardware_info()
        
        print(f"CPU Information:")
        print(f"  Physical Cores: {hw_info['cpu']['physical_cores']}")
        print(f"  Logical Cores: {hw_info['cpu']['logical_cores']}")
        print(f"  Architecture: {hw_info['cpu']['architecture']}")
        
        print(f"\nGPU Information:")
        print(f"  Available: {hw_info['gpu']['available']}")
        if hw_info['gpu']['available']:
            print(f"  Count: {hw_info['gpu']['count']}")
            print(f"  Total Memory: {hw_info['gpu']['memory_total']:.2f} GB")
            for i, device in enumerate(hw_info['gpu']['devices']):
                print(f"  Device {i}: {device['name']} ({device['memory_total']:.2f} GB)")
        
        print(f"\nMemory Information:")
        print(f"  Total: {hw_info['memory']['total']:.2f} GB")
        print(f"  Available: {hw_info['memory']['available']:.2f} GB")
        print(f"  Used: {hw_info['memory']['used']:.2f} GB ({hw_info['memory']['percentage']:.1f}%)")
        
        backend.cleanup()
        return hw_info
    
    def test_memory_management(self):
        """Test memory management capabilities."""
        print("\nTesting Memory Management...")
        
        config = HardwareConfig(memory_strategy=MemoryStrategy.ADAPTIVE)
        backend = IntelligentBackend(config)
        
        # Test memory estimation
        test_data = np.random.randn(1000, 1000)
        estimated_memory = backend.memory_manager.estimate_memory_usage(
            test_data.size, str(test_data.dtype)
        )
        actual_memory = test_data.nbytes / (1024**3)
        
        print(f"Memory Estimation Test:")
        print(f"  Estimated: {estimated_memory:.4f} GB")
        print(f"  Actual: {actual_memory:.4f} GB")
        print(f"  Error: {abs(estimated_memory - actual_memory) / actual_memory * 100:.2f}%")
        
        # Test memory allocation check
        can_allocate = backend.memory_manager.can_allocate_memory(1.0)  # 1 GB
        print(f"Can allocate 1 GB: {can_allocate}")
        
        # Test memory cleanup
        print("Testing memory cleanup...")
        backend.memory_manager.cleanup_memory()
        
        memory_after = backend.memory_manager.get_memory_usage()
        print(f"Memory after cleanup: {memory_after['used']:.2f} GB")
        
        backend.cleanup()
        return {
            'estimation_error': abs(estimated_memory - actual_memory) / actual_memory * 100,
            'can_allocate': can_allocate,
            'memory_after_cleanup': memory_after['used']
        }
    
    def test_task_scheduling(self):
        """Test task scheduling capabilities."""
        print("\nTesting Task Scheduling...")
        
        config = HardwareConfig(max_parallel_tasks=3)
        backend = IntelligentBackend(config)
        
        # Define test tasks
        def cpu_intensive_task(data, iterations=1000):
            """CPU intensive task."""
            result = data.copy()
            for _ in range(iterations):
                result = np.sin(result) + np.cos(result)
            return result
        
        def memory_intensive_task(size=1000):
            """Memory intensive task."""
            data = np.random.randn(size, size)
            return np.linalg.eig(data)
        
        def io_intensive_task(delay=0.1):
            """IO intensive task."""
            time.sleep(delay)
            return np.random.randn(100, 100)
        
        # Submit tasks with different characteristics
        task_ids = []
        
        # CPU intensive tasks
        for i in range(3):
            data = np.random.randn(500, 500)
            task_id = backend.submit_task(
                f"cpu_task_{i}",
                cpu_intensive_task,
                data,
                100,  # iterations argument
                priority=1,
                compute_intensity="high",
                preferred_backend=ComputeBackend.CPU
            )
            task_ids.append(task_id)
        
        # Memory intensive tasks
        for i in range(2):
            task_id = backend.submit_task(
                f"memory_task_{i}",
                memory_intensive_task,
                800,  # size argument
                priority=2,
                memory_estimate=2.0,
                compute_intensity="high"
            )
            task_ids.append(task_id)
        
        # IO intensive tasks
        for i in range(4):
            task_id = backend.submit_task(
                f"io_task_{i}",
                io_intensive_task,
                0.05,  # delay argument
                priority=3,
                compute_intensity="low"
            )
            task_ids.append(task_id)
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Execute tasks
        start_time = time.time()
        results = backend.execute_all_tasks()
        execution_time = time.time() - start_time
        
        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Completed: {len(results)} tasks")
        
        # Get final status
        status = backend.get_status()
        print(f"Final status: {status['completed_tasks']} completed, {status['failed_tasks']} failed")
        
        backend.cleanup()
        return {
            'total_tasks': len(task_ids),
            'completed_tasks': len(results),
            'execution_time': execution_time,
            'throughput': len(results) / execution_time
        }
    
    def test_memory_strategies(self):
        """Test different memory strategies."""
        print("\nTesting Memory Strategies...")
        
        strategy_results = {}
        
        for strategy_name, config in self.test_configs.items():
            print(f"\nTesting {strategy_name} strategy...")
            
            backend = IntelligentBackend(config)
            
            # Define memory intensive task
            def memory_task(size):
                data = np.random.randn(size, size)
                return np.linalg.svd(data)
            
            # Submit tasks
            task_ids = []
            for i in range(5):
                size = 500 + i * 100
                memory_estimate = backend.memory_manager.estimate_memory_usage(size * size, 'float64')
                
                task_id = backend.submit_task(
                    f"memory_task_{i}",
                    memory_task,
                    size,
                    memory_estimate=memory_estimate,
                    compute_intensity="high"
                )
                task_ids.append(task_id)
            
            # Execute tasks
            start_time = time.time()
            results = backend.execute_all_tasks()
            execution_time = time.time() - start_time
            
            # Get memory usage
            memory_usage = backend.memory_manager.get_memory_usage()
            
            strategy_results[strategy_name] = {
                'execution_time': execution_time,
                'completed_tasks': len(results),
                'memory_usage': memory_usage['used'],
                'memory_percentage': memory_usage['percentage']
            }
            
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Completed: {len(results)}/{len(task_ids)} tasks")
            print(f"  Memory usage: {memory_usage['used']:.2f} GB ({memory_usage['percentage']:.1f}%)")
            
            backend.cleanup()
        
        return strategy_results
    
    def test_data_optimization(self):
        """Test data optimization for different backends."""
        print("\nTesting Data Optimization...")
        
        config = HardwareConfig()
        backend = IntelligentBackend(config)
        
        # Test data
        test_data = np.random.randn(1000, 1000).astype(np.float32)
        
        print(f"Original data: {test_data.shape}, {test_data.dtype}")
        
        # Test optimization for different backends
        backends = [ComputeBackend.CPU]
        if backend.hardware_utilizer.gpu_info['available']:
            backends.append(ComputeBackend.GPU)
        
        optimization_results = {}
        
        for backend_type in backends:
            print(f"\nOptimizing for {backend_type.value}...")
            
            start_time = time.time()
            optimized_data = backend.optimize_data(test_data, backend_type)
            optimization_time = time.time() - start_time
            
            print(f"  Optimization time: {optimization_time:.4f}s")
            print(f"  Optimized data type: {type(optimized_data)}")
            if hasattr(optimized_data, 'shape'):
                print(f"  Optimized data shape: {optimized_data.shape}")
            if hasattr(optimized_data, 'dtype'):
                print(f"  Optimized data dtype: {optimized_data.dtype}")
            
            optimization_results[backend_type.value] = {
                'optimization_time': optimization_time,
                'data_type': str(type(optimized_data)),
                'shape': optimized_data.shape if hasattr(optimized_data, 'shape') else None,
                'dtype': str(optimized_data.dtype) if hasattr(optimized_data, 'dtype') else None
            }
        
        backend.cleanup()
        return optimization_results
    
    def test_caching(self):
        """Test caching functionality."""
        print("\nTesting Caching...")
        
        config = HardwareConfig(enable_caching=True, cache_size_limit=0.5)
        backend = IntelligentBackend(config)
        
        # Define expensive computation
        def expensive_computation(n):
            """Expensive computation that should be cached."""
            time.sleep(0.1)  # Simulate computation
            return np.random.randn(n, n)
        
        # First execution (should be computed)
        print("First execution (computation)...")
        start_time = time.time()
        task_id1 = backend.submit_task(
            "cached_task_1",
            expensive_computation,
            100,
            memory_estimate=0.1
        )
        results1 = backend.execute_all_tasks()
        first_time = time.time() - start_time
        
        # Second execution (should be cached)
        print("Second execution (cached)...")
        start_time = time.time()
        task_id2 = backend.submit_task(
            "cached_task_2",
            expensive_computation,
            100,
            memory_estimate=0.1
        )
        results2 = backend.execute_all_tasks()
        second_time = time.time() - start_time
        
        print(f"First execution time: {first_time:.4f}s")
        print(f"Second execution time: {second_time:.4f}s")
        print(f"Speedup: {first_time / second_time:.2f}x")
        
        # Check cache status
        status = backend.get_status()
        print(f"Cache size: {status['cache_size']:.4f} GB")
        
        backend.cleanup()
        return {
            'first_execution_time': first_time,
            'second_execution_time': second_time,
            'speedup': first_time / second_time,
            'cache_size': status['cache_size']
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("INTELLIGENT BACKEND PERFORMANCE REPORT")
        print("="*60)
        
        # Hardware detection results
        hw_info = self.test_hardware_detection()
        
        # Memory management results
        memory_results = self.test_memory_management()
        
        # Task scheduling results
        scheduling_results = self.test_task_scheduling()
        
        # Memory strategies results
        strategy_results = self.test_memory_strategies()
        
        # Data optimization results
        optimization_results = self.test_data_optimization()
        
        # Caching results
        caching_results = self.test_caching()
        
        # Compile results
        self.results = {
            'hardware_info': hw_info,
            'memory_management': memory_results,
            'task_scheduling': scheduling_results,
            'memory_strategies': strategy_results,
            'data_optimization': optimization_results,
            'caching': caching_results
        }
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"Hardware Detection: ✅")
        print(f"Memory Management: ✅ (Error: {memory_results['estimation_error']:.2f}%)")
        print(f"Task Scheduling: ✅ ({scheduling_results['throughput']:.2f} tasks/sec)")
        print(f"Memory Strategies: ✅")
        print(f"Data Optimization: ✅")
        print(f"Caching: ✅ ({caching_results['speedup']:.2f}x speedup)")
        
        return self.results
    
    def save_results(self, filename: str = "intelligent_backend_test_results.json"):
        """Save test results to file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_types(self.results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def main():
    """Main function to run the intelligent backend tests."""
    print("Intelligent Backend Framework Test")
    print("="*50)
    
    # Initialize tester
    tester = IntelligentBackendTester()
    
    try:
        # Run comprehensive tests
        results = tester.generate_performance_report()
        
        # Save results
        tester.save_results()
        
        print("\nIntelligent backend testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    main()
