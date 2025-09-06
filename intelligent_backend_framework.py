#!/usr/bin/env python3
"""
Intelligent Backend Framework for LRDBenchmark

This module provides sophisticated hardware utilization strategies, memory-aware
computation scheduling, and distributed computing support for the LRDBenchmark framework.

Author: LRDBenchmark Team
Date: 2025-01-05
"""

import os
import psutil
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
import time
import logging
import queue
import gc
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import subprocess
import platform
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU libraries
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    GPU = "gpu"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"

class MemoryStrategy(Enum):
    """Memory management strategies."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"

@dataclass
class HardwareConfig:
    """Hardware configuration for intelligent backend."""
    # CPU configuration
    max_cpu_cores: int = None
    cpu_memory_limit: float = None  # GB
    
    # GPU configuration
    gpu_memory_limit: float = None  # GB
    gpu_memory_fraction: float = 0.8
    
    # Distributed computing
    enable_distributed: bool = False
    distributed_backend: str = "dask"  # "dask" or "ray"
    cluster_nodes: int = None
    
    # Memory management
    memory_strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.8
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_parallel_tasks: int = None
    enable_caching: bool = True
    cache_size_limit: float = 1.0  # GB
    
    def __post_init__(self):
        if self.max_cpu_cores is None:
            self.max_cpu_cores = mp.cpu_count()
        
        if self.cpu_memory_limit is None:
            self.cpu_memory_limit = psutil.virtual_memory().total / (1024**3) * 0.8
        
        if self.max_parallel_tasks is None:
            self.max_parallel_tasks = min(self.max_cpu_cores, 8)

@dataclass
class TaskInfo:
    """Information about a computation task."""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    memory_estimate: float = 0.0  # GB
    compute_intensity: str = "medium"  # "low", "medium", "high"
    preferred_backend: ComputeBackend = ComputeBackend.CPU
    timeout: float = None
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class MemoryManager:
    """Memory-aware computation scheduling and management."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.memory_usage = 0.0
        self.memory_history = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.config.enable_memory_monitoring and not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Background memory monitoring."""
        while self.monitoring:
            try:
                current_memory = psutil.virtual_memory().used / (1024**3)
                self.memory_usage = current_memory
                self.memory_history.append((time.time(), current_memory))
                
                # Keep only last 100 measurements
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # Check if cleanup is needed
                if current_memory / (psutil.virtual_memory().total / (1024**3)) > self.config.memory_cleanup_threshold:
                    self.cleanup_memory()
                
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(5.0)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),
            'available': memory.available / (1024**3),
            'used': memory.used / (1024**3),
            'percentage': memory.percent,
            'cached': getattr(memory, 'cached', 0) / (1024**3),
            'buffers': getattr(memory, 'buffers', 0) / (1024**3)
        }
    
    def estimate_memory_usage(self, data_size: int, dtype: str = 'float64') -> float:
        """Estimate memory usage for given data size and type."""
        dtype_sizes = {
            'float32': 4,
            'float64': 8,
            'int32': 4,
            'int64': 8,
            'complex64': 8,
            'complex128': 16
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 8)
        return (data_size * bytes_per_element) / (1024**3)
    
    def can_allocate_memory(self, required_memory: float) -> bool:
        """Check if required memory can be allocated."""
        current_usage = self.get_memory_usage()
        available_memory = current_usage['available']
        return available_memory >= required_memory
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory usage after cleanup
        memory_after = self.get_memory_usage()
        logger.info(f"Memory after cleanup: {memory_after['used']:.2f} GB used, {memory_after['available']:.2f} GB available")
    
    def get_memory_strategy(self) -> MemoryStrategy:
        """Get current memory strategy based on usage."""
        if self.config.memory_strategy == MemoryStrategy.ADAPTIVE:
            current_usage = self.get_memory_usage()
            if current_usage['percentage'] > 80:
                return MemoryStrategy.CONSERVATIVE
            elif current_usage['percentage'] < 50:
                return MemoryStrategy.AGGRESSIVE
            else:
                return MemoryStrategy.ADAPTIVE
        else:
            return self.config.memory_strategy

class HardwareUtilizer:
    """Sophisticated hardware utilization strategies."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.cpu_cores = self._detect_cpu_cores()
        self.gpu_info = self._detect_gpu_info()
        self.memory_manager = MemoryManager(config)
        
    def _detect_cpu_cores(self) -> Dict[str, Any]:
        """Detect CPU information and capabilities."""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information and capabilities."""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'memory_total': 0,
            'memory_available': 0
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['count'] = torch.cuda.device_count()
            
            for i in range(gpu_info['count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'memory_available': torch.cuda.memory_reserved(i) / (1024**3)
                }
                gpu_info['devices'].append(device_info)
                gpu_info['memory_total'] += device_info['memory_total']
                gpu_info['memory_available'] += device_info['memory_available']
        
        return gpu_info
    
    def get_optimal_backend(self, task_info: TaskInfo) -> ComputeBackend:
        """Determine optimal compute backend for a task."""
        # Check if GPU is preferred and available
        if (task_info.preferred_backend == ComputeBackend.GPU and 
            self.gpu_info['available'] and 
            task_info.compute_intensity in ['high', 'medium']):
            return ComputeBackend.GPU
        
        # Check if distributed computing is needed
        if (task_info.memory_estimate > self.config.cpu_memory_limit * 0.5 or
            task_info.compute_intensity == 'high'):
            return ComputeBackend.DISTRIBUTED
        
        # Default to CPU
        return ComputeBackend.CPU
    
    def optimize_for_hardware(self, data: np.ndarray, backend: ComputeBackend) -> np.ndarray:
        """Optimize data for specific hardware backend."""
        if backend == ComputeBackend.GPU and TORCH_AVAILABLE:
            # Convert to PyTorch tensor and move to GPU
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data)
            return data.cuda()
        
        elif backend == ComputeBackend.CPU:
            # Ensure data is in optimal format for CPU
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            return np.ascontiguousarray(data)
        
        return data
    
    def get_parallel_config(self, task_count: int) -> Dict[str, Any]:
        """Get optimal parallel processing configuration."""
        max_workers = min(task_count, self.config.max_parallel_tasks)
        
        # Adjust based on memory usage
        memory_usage = self.memory_manager.get_memory_usage()
        if memory_usage['percentage'] > 70:
            max_workers = max(1, max_workers // 2)
        
        return {
            'max_workers': max_workers,
            'chunk_size': max(1, task_count // max_workers),
            'backend': 'thread' if task_count < 10 else 'process'
        }

class DistributedComputing:
    """Distributed computing support using Dask or Ray."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.client = None
        self.cluster = None
        self.backend = config.distributed_backend
        
    def initialize_cluster(self):
        """Initialize distributed computing cluster."""
        if not DASK_AVAILABLE and not RAY_AVAILABLE:
            logger.warning("No distributed computing backend available")
            return False
        
        try:
            if self.backend == "dask" and DASK_AVAILABLE:
                self._initialize_dask_cluster()
            elif self.backend == "ray" and RAY_AVAILABLE:
                self._initialize_ray_cluster()
            else:
                logger.warning(f"Backend {self.backend} not available")
                return False
            
            logger.info(f"Distributed computing cluster initialized with {self.backend}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cluster: {e}")
            return False
    
    def _initialize_dask_cluster(self):
        """Initialize Dask cluster."""
        if self.config.cluster_nodes:
            # Multi-node cluster (requires external setup)
            self.client = Client("scheduler-address:8786")
        else:
            # Local cluster
            self.cluster = LocalCluster(
                n_workers=self.config.max_cpu_cores,
                threads_per_worker=2,
                memory_limit=f"{self.config.cpu_memory_limit:.1f}GB"
            )
            self.client = Client(self.cluster)
    
    def _initialize_ray_cluster(self):
        """Initialize Ray cluster."""
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.config.max_cpu_cores,
                memory=int(self.config.cpu_memory_limit * 1024**3),
                ignore_reinit_error=True
            )
    
    def shutdown_cluster(self):
        """Shutdown distributed computing cluster."""
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()
            logger.info("Distributed computing cluster shutdown")
        except Exception as e:
            logger.error(f"Error shutting down cluster: {e}")
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to distributed cluster."""
        if self.backend == "dask" and self.client:
            return self.client.submit(func, *args, **kwargs)
        elif self.backend == "ray" and RAY_AVAILABLE:
            return ray.remote(func).remote(*args, **kwargs)
        else:
            raise RuntimeError("Distributed computing not initialized")

class TaskScheduler:
    """Intelligent task scheduling with memory awareness."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.hardware_utilizer = HardwareUtilizer(config)
        self.memory_manager = MemoryManager(config)
        self.distributed_computing = DistributedComputing(config)
        
        self.task_queue = queue.PriorityQueue()
        self._task_counter = 0
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        self.cache = {}
        self.cache_size = 0
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        # Initialize distributed computing if enabled
        if config.enable_distributed:
            self.distributed_computing.initialize_cluster()
    
    def submit_task(self, task_info: TaskInfo) -> str:
        """Submit a task for execution."""
        # Check if task is already cached
        if self.config.enable_caching:
            cache_key = self._get_cache_key(task_info)
            if cache_key in self.cache:
                logger.info(f"Task {task_info.task_id} found in cache")
                self.completed_tasks[task_info.task_id] = self.cache[cache_key]
                return task_info.task_id
        
        # Check memory requirements
        if not self.memory_manager.can_allocate_memory(task_info.memory_estimate):
            logger.warning(f"Insufficient memory for task {task_info.task_id}")
            self.failed_tasks[task_info.task_id] = "Insufficient memory"
            return task_info.task_id
        
        # Add to task queue with counter to ensure unique ordering
        self._task_counter += 1
        self.task_queue.put((task_info.priority, self._task_counter, task_info))
        logger.info(f"Task {task_info.task_id} submitted to queue")
        
        return task_info.task_id
    
    def execute_tasks(self, max_concurrent: int = None) -> Dict[str, Any]:
        """Execute tasks from the queue."""
        if max_concurrent is None:
            max_concurrent = self.config.max_parallel_tasks
        
        results = {}
        active_tasks = {}
        
        try:
            while not self.task_queue.empty() or active_tasks:
                # Start new tasks if we have capacity
                while len(active_tasks) < max_concurrent and not self.task_queue.empty():
                    try:
                        _, _, task_info = self.task_queue.get_nowait()
                        self._start_task(task_info, active_tasks)
                    except queue.Empty:
                        break
                
                # Check for completed tasks
                completed_tasks = []
                for task_id, future in active_tasks.items():
                    if future.done():
                        completed_tasks.append(task_id)
                
                # Process completed tasks
                for task_id in completed_tasks:
                    future = active_tasks.pop(task_id)
                    try:
                        result = future.result()
                        self.completed_tasks[task_id] = result
                        results[task_id] = result
                        
                        # Cache result if enabled
                        if self.config.enable_caching:
                            self._cache_result(task_id, result)
                        
                        logger.info(f"Task {task_id} completed successfully")
                    except Exception as e:
                        self.failed_tasks[task_id] = str(e)
                        logger.error(f"Task {task_id} failed: {e}")
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Task execution interrupted by user")
        
        return results
    
    def _start_task(self, task_info: TaskInfo, active_tasks: Dict[str, Any]):
        """Start execution of a task."""
        # Determine optimal backend
        backend = self.hardware_utilizer.get_optimal_backend(task_info)
        
        # Get parallel configuration
        parallel_config = self.hardware_utilizer.get_parallel_config(1)
        
        # Choose execution method
        if backend == ComputeBackend.DISTRIBUTED and self.distributed_computing.client:
            future = self.distributed_computing.submit_task(
                task_info.function, *task_info.args, **task_info.kwargs
            )
        elif parallel_config['backend'] == 'process':
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(task_info.function, *task_info.args, **task_info.kwargs)
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(task_info.function, *task_info.args, **task_info.kwargs)
        
        active_tasks[task_info.task_id] = future
        self.running_tasks[task_info.task_id] = task_info
    
    def _get_cache_key(self, task_info: TaskInfo) -> str:
        """Generate cache key for task."""
        import hashlib
        
        # Create hash of function and arguments
        key_data = {
            'function': task_info.function.__name__,
            'args': task_info.args,
            'kwargs': task_info.kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_result(self, task_id: str, result: Any):
        """Cache task result."""
        if not self.config.enable_caching:
            return
        
        # Estimate result size
        try:
            if isinstance(result, np.ndarray):
                result_size = result.nbytes / (1024**3)
            else:
                result_size = len(pickle.dumps(result)) / (1024**3)
        except:
            result_size = 0.1  # Default estimate
        
        # Check cache size limit
        if self.cache_size + result_size > self.config.cache_size_limit:
            self._cleanup_cache()
        
        # Add to cache
        cache_key = self._get_cache_key(self.running_tasks[task_id])
        self.cache[cache_key] = result
        self.cache_size += result_size
    
    def _cleanup_cache(self):
        """Cleanup cache to free memory."""
        # Remove oldest 25% of cache entries
        entries_to_remove = len(self.cache) // 4
        for _ in range(entries_to_remove):
            if self.cache:
                key = next(iter(self.cache))
                del self.cache[key]
        
        # Force garbage collection
        gc.collect()
        self.cache_size = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'cache_size': self.cache_size,
            'memory_usage': self.memory_manager.get_memory_usage(),
            'hardware_info': {
                'cpu_cores': self.hardware_utilizer.cpu_cores,
                'gpu_info': self.hardware_utilizer.gpu_info
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.memory_manager.stop_monitoring()
        self.distributed_computing.shutdown_cluster()
        logger.info("Task scheduler cleanup completed")

class IntelligentBackend:
    """Main intelligent backend interface."""
    
    def __init__(self, config: HardwareConfig = None):
        self.config = config or HardwareConfig()
        self.scheduler = TaskScheduler(self.config)
        self.hardware_utilizer = self.scheduler.hardware_utilizer
        self.memory_manager = self.scheduler.memory_manager
        
        logger.info(f"Intelligent backend initialized with {self.config.max_cpu_cores} CPU cores")
        if self.hardware_utilizer.gpu_info['available']:
            logger.info(f"GPU support enabled: {self.hardware_utilizer.gpu_info['count']} devices")
    
    def submit_task(self, task_id: str, function: Callable, *args, 
                   priority: int = 0, memory_estimate: float = 0.0,
                   compute_intensity: str = "medium", preferred_backend: ComputeBackend = ComputeBackend.CPU,
                   timeout: float = None, **kwargs) -> str:
        """Submit a task for execution."""
        task_info = TaskInfo(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            memory_estimate=memory_estimate,
            compute_intensity=compute_intensity,
            preferred_backend=preferred_backend,
            timeout=timeout
        )
        
        return self.scheduler.submit_task(task_info)
    
    def execute_all_tasks(self, max_concurrent: int = None) -> Dict[str, Any]:
        """Execute all submitted tasks."""
        return self.scheduler.execute_tasks(max_concurrent)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information."""
        return {
            'cpu': self.hardware_utilizer.cpu_cores,
            'gpu': self.hardware_utilizer.gpu_info,
            'memory': self.memory_manager.get_memory_usage(),
            'config': {
                'max_cpu_cores': self.config.max_cpu_cores,
                'cpu_memory_limit': self.config.cpu_memory_limit,
                'gpu_memory_limit': self.config.gpu_memory_limit,
                'distributed_enabled': self.config.enable_distributed,
                'memory_strategy': self.config.memory_strategy.value
            }
        }
    
    def optimize_data(self, data: np.ndarray, backend: ComputeBackend = None) -> np.ndarray:
        """Optimize data for specific hardware backend."""
        if backend is None:
            # Auto-detect optimal backend
            memory_estimate = self.memory_manager.estimate_memory_usage(data.size, str(data.dtype))
            task_info = TaskInfo(
                task_id="optimize",
                function=lambda x: x,
                memory_estimate=memory_estimate,
                compute_intensity="low"
            )
            backend = self.hardware_utilizer.get_optimal_backend(task_info)
        
        return self.hardware_utilizer.optimize_for_hardware(data, backend)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current backend status."""
        return self.scheduler.get_status()
    
    def cleanup(self):
        """Cleanup all resources."""
        self.scheduler.cleanup()
        logger.info("Intelligent backend cleanup completed")

def main():
    """Example usage of the intelligent backend framework."""
    print("Intelligent Backend Framework Example")
    print("="*50)
    
    # Create configuration
    config = HardwareConfig(
        max_cpu_cores=4,
        cpu_memory_limit=8.0,
        enable_distributed=False,
        memory_strategy=MemoryStrategy.ADAPTIVE,
        enable_caching=True
    )
    
    # Initialize backend
    backend = IntelligentBackend(config)
    
    # Get hardware info
    hw_info = backend.get_hardware_info()
    print(f"CPU Cores: {hw_info['cpu']['logical_cores']}")
    print(f"GPU Available: {hw_info['gpu']['available']}")
    print(f"Memory: {hw_info['memory']['total']:.2f} GB total, {hw_info['memory']['available']:.2f} GB available")
    
    # Example task function
    def compute_task(data, multiplier=2.0):
        """Example computation task."""
        time.sleep(0.1)  # Simulate computation
        return data * multiplier
    
    # Submit tasks
    task_ids = []
    for i in range(5):
        data = np.random.randn(1000, 100)
        memory_estimate = backend.memory_manager.estimate_memory_usage(data.size, str(data.dtype))
        
        task_id = backend.submit_task(
            f"task_{i}",
            compute_task,
            data,
            float(i + 1),  # multiplier argument
            memory_estimate=memory_estimate,
            compute_intensity="medium"
        )
        task_ids.append(task_id)
    
    # Execute tasks
    print(f"\nExecuting {len(task_ids)} tasks...")
    results = backend.execute_all_tasks()
    
    # Print results
    print(f"Completed {len(results)} tasks")
    for task_id, result in results.items():
        print(f"  {task_id}: {result.shape if hasattr(result, 'shape') else type(result)}")
    
    # Get final status
    status = backend.get_status()
    print(f"\nFinal Status:")
    print(f"  Completed: {status['completed_tasks']}")
    print(f"  Failed: {status['failed_tasks']}")
    print(f"  Memory Usage: {status['memory_usage']['used']:.2f} GB")
    
    # Cleanup
    backend.cleanup()
    print("\nIntelligent backend example completed!")

if __name__ == "__main__":
    main()
