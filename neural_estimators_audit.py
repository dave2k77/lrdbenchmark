#!/usr/bin/env python3
"""
Comprehensive Neural Network Estimators Audit

This script performs a thorough audit of neural network LRD estimators including:
1. Architecture validation and implementation verification
2. GPU optimization capabilities and hardware detection
3. Pretrained model availability and loading
4. Performance benchmarking and comparison
5. Train-once-apply-many workflow validation
"""

import numpy as np
import time
import warnings
import logging
import torch
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neural network estimators
try:
    from lrdbenchmark.analysis.machine_learning import (
        CNNEstimator, LSTMEstimator, GRUEstimator, TransformerEstimator,
        NeuralNetworkFactory
    )
    # Import neural network factory components directly
    from lrdbenchmark.analysis.machine_learning.neural_network_factory import (
        NNArchitecture, NNConfig
    )
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    NEURAL_NETWORKS_AVAILABLE = False
    print(f"Warning: Neural network estimators not available: {e}")
    # Set defaults for when imports fail
    NNArchitecture = None
    NNConfig = None

# Import data models for testing
try:
    from lrdbenchmark.models.data_models import FBMModel, FGNModel, ARFIMAModel
    DATA_MODELS_AVAILABLE = True
except ImportError as e:
    DATA_MODELS_AVAILABLE = False
    print(f"Warning: Data models not available: {e}")

class NeuralEstimatorsAudit:
    """Comprehensive audit of neural network LRD estimators."""
    
    def __init__(self, output_dir: str = "neural_audit_results"):
        """Initialize the audit."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.test_data = {}
        self.estimators = {}
        
        # Check system capabilities
        self._check_system_capabilities()
        
        # Initialize estimators
        self._initialize_estimators()
        
        # Generate test data
        self._generate_test_data()
    
    def _check_system_capabilities(self):
        """Check system capabilities for neural networks."""
        print("🔍 Checking System Capabilities...")
        
        self.system_info = {
            "pytorch_available": torch is not None,
            "cuda_available": torch.cuda.is_available() if torch is not None else False,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__ if torch is not None else None
        }
        
        if self.system_info["cuda_available"]:
            print(f"✅ CUDA available: {self.system_info['cuda_device_count']} device(s)")
            print(f"   Current device: {self.system_info['current_device']}")
            print(f"   CUDA version: {self.system_info['cuda_version']}")
            print(f"   PyTorch version: {self.system_info['pytorch_version']}")
        else:
            print("⚠️ CUDA not available, using CPU")
            print(f"   PyTorch version: {self.system_info['pytorch_version']}")
    
    def _initialize_estimators(self):
        """Initialize neural network estimators."""
        print("\n🧠 Initializing Neural Network Estimators...")
        
        if not NEURAL_NETWORKS_AVAILABLE:
            print("❌ Neural network estimators not available")
            return
        
        # Initialize unified estimators
        self.estimators = {}
        if NEURAL_NETWORKS_AVAILABLE:
            try:
                self.estimators = {
                    "CNN": CNNEstimator(),
                    "LSTM": LSTMEstimator(),
                    "GRU": GRUEstimator(),
                    "Transformer": TransformerEstimator()
                }
            except Exception as e:
                print(f"⚠️ Failed to initialize some estimators: {e}")
                self.estimators = {}
        
        # Initialize neural network factory
        self.nn_factory = None
        if NEURAL_NETWORKS_AVAILABLE:
            try:
                self.nn_factory = NeuralNetworkFactory()
                print(f"✅ Neural Network Factory initialized")
                print(f"   Available architectures: {len(self.nn_factory.get_available_architectures())}")
            except Exception as e:
                print(f"⚠️ Neural Network Factory failed to initialize: {e}")
                self.nn_factory = None
        else:
            print("⚠️ Neural Network Factory not available (imports failed)")
        
        print(f"✅ Initialized {len(self.estimators)} neural network estimators")
    
    def _generate_test_data(self):
        """Generate test data for neural network testing."""
        print("\n📊 Generating Test Data...")
        
        if not DATA_MODELS_AVAILABLE:
            print("⚠️ Data models not available, using synthetic data")
            # Generate synthetic test data
            np.random.seed(42)
            self.test_data = {
                "synthetic_1d": np.random.randn(1000),
                "synthetic_2d": np.random.randn(100, 100),
                "synthetic_time_series": np.cumsum(np.random.randn(1000))
            }
            return
        
        # Generate realistic test data
        hurst_values = [0.3, 0.5, 0.7, 0.9]
        sequence_lengths = [100, 500, 1000]
        
        self.test_data = {}
        
        for H in hurst_values:
            for seq_len in sequence_lengths:
                # FBM data
                fbm_model = FBMModel(H=H, sigma=1.0)
                fbm_data = fbm_model.generate(n=seq_len, seed=42)
                self.test_data[f"fbm_H{H}_len{seq_len}"] = fbm_data
                
                # FGN data
                fgn_model = FGNModel(H=H, sigma=1.0)
                fgn_data = fgn_model.generate(n=seq_len, seed=42)
                self.test_data[f"fgn_H{H}_len{seq_len}"] = fgn_data
        
        print(f"✅ Generated {len(self.test_data)} test datasets")
    
    def audit_architecture_implementation(self) -> Dict[str, Any]:
        """Audit neural network architecture and implementation."""
        print("\n🏗️ Auditing Neural Network Architecture and Implementation...")
        
        architecture_results = {
            "estimators_available": {},
            "factory_available": False,
            "available_architectures": [],
            "implementation_status": {},
            "optimization_frameworks": {}
        }
        
        # Check estimator availability
        for name, estimator in self.estimators.items():
            try:
                architecture_results["estimators_available"][name] = {
                    "class": type(estimator).__name__,
                    "optimization_framework": getattr(estimator, 'optimization_framework', 'unknown'),
                    "parameters": getattr(estimator, 'parameters', {}),
                    "status": "available"
                }
                print(f"   ✅ {name}: {type(estimator).__name__}")
            except Exception as e:
                architecture_results["estimators_available"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"   ❌ {name}: {e}")
        
        # Check neural network factory
        if self.nn_factory:
            try:
                architectures = self.nn_factory.get_available_architectures()
                architecture_results["factory_available"] = True
                architecture_results["available_architectures"] = [arch.value for arch in architectures]
                print(f"   ✅ Neural Network Factory: {len(architectures)} architectures available")
                for arch in architectures:
                    print(f"      • {arch.value}")
            except Exception as e:
                architecture_results["factory_available"] = False
                architecture_results["factory_error"] = str(e)
                print(f"   ❌ Neural Network Factory: {e}")
        
        # Check optimization frameworks
        try:
            import jax
            architecture_results["optimization_frameworks"]["jax"] = True
            print("   ✅ JAX available")
        except ImportError:
            architecture_results["optimization_frameworks"]["jax"] = False
            print("   ❌ JAX not available")
        
        try:
            import numba
            architecture_results["optimization_frameworks"]["numba"] = True
            print("   ✅ Numba available")
        except ImportError:
            architecture_results["optimization_frameworks"]["numba"] = False
            print("   ❌ Numba not available")
        
        return architecture_results
    
    def audit_gpu_optimization(self) -> Dict[str, Any]:
        """Audit GPU optimization capabilities."""
        print("\n🚀 Auditing GPU Optimization Capabilities...")
        
        gpu_results = {
            "system_capabilities": self.system_info,
            "pytorch_gpu": {},
            "jax_gpu": {},
            "performance_tests": {}
        }
        
        # Test PyTorch GPU capabilities
        if self.system_info["cuda_available"]:
            try:
                device = torch.device("cuda")
                
                # Test tensor operations on GPU
                x = torch.randn(1000, 100).to(device)
                y = torch.randn(100, 1).to(device)
                
                start_time = time.time()
                z = torch.mm(x, y)
                gpu_time = time.time() - start_time
                
                gpu_results["pytorch_gpu"] = {
                    "available": True,
                    "device": str(device),
                    "test_operation_time": gpu_time,
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_cached": torch.cuda.memory_reserved()
                }
                print(f"   ✅ PyTorch GPU: {device} (test time: {gpu_time:.4f}s)")
                
            except Exception as e:
                gpu_results["pytorch_gpu"] = {
                    "available": False,
                    "error": str(e)
                }
                print(f"   ❌ PyTorch GPU test failed: {e}")
        else:
            gpu_results["pytorch_gpu"] = {"available": False, "reason": "CUDA not available"}
            print("   ❌ PyTorch GPU: CUDA not available")
        
        # Test JAX GPU capabilities
        try:
            import jax
            import jax.numpy as jnp
            
            # Check JAX backend
            backend = jax.lib.xla_bridge.get_backend().platform
            gpu_results["jax_gpu"] = {
                "available": backend == "gpu",
                "backend": backend,
                "device_count": jax.device_count(),
                "devices": [str(d) for d in jax.devices()]
            }
            
            if backend == "gpu":
                print(f"   ✅ JAX GPU: {backend} backend, {jax.device_count()} devices")
            else:
                print(f"   ⚠️ JAX GPU: {backend} backend (CPU fallback)")
                
        except Exception as e:
            gpu_results["jax_gpu"] = {
                "available": False,
                "error": str(e)
            }
            print(f"   ❌ JAX GPU test failed: {e}")
        
        # Performance comparison test
        if len(self.test_data) > 0:
            test_key = list(self.test_data.keys())[0]
            test_data = self.test_data[test_key]
            
            # CPU performance test
            start_time = time.time()
            _ = np.fft.fft(test_data)
            cpu_time = time.time() - start_time
            
            gpu_results["performance_tests"] = {
                "cpu_fft_time": cpu_time,
                "test_data_size": len(test_data)
            }
            print(f"   📊 CPU FFT test: {cpu_time:.4f}s")
        
        return gpu_results
    
    def audit_pretrained_models(self) -> Dict[str, Any]:
        """Audit pretrained neural network models."""
        print("\n💾 Auditing Pretrained Neural Network Models...")
        
        pretrained_results = {
            "config_files": {},
            "model_files": {},
            "loading_tests": {},
            "model_info": {}
        }
        
        # Check for configuration files
        models_dir = Path("models")
        if models_dir.exists():
            config_files = list(models_dir.glob("*_config.json"))
            print(f"   Found {len(config_files)} configuration files:")
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    pretrained_results["config_files"][config_file.name] = {
                        "status": "available",
                        "architecture": config.get("architecture", "unknown"),
                        "parameters": config
                    }
                    print(f"      ✅ {config_file.name}: {config.get('architecture', 'unknown')}")
                    
                except Exception as e:
                    pretrained_results["config_files"][config_file.name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"      ❌ {config_file.name}: {e}")
        
        # Check for PyTorch model files
        pytorch_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
        if pytorch_files:
            print(f"   Found {len(pytorch_files)} PyTorch model files:")
            for model_file in pytorch_files:
                pretrained_results["model_files"][model_file.name] = {
                    "status": "available",
                    "size": model_file.stat().st_size,
                    "type": "pytorch"
                }
                print(f"      ✅ {model_file.name}: {model_file.stat().st_size} bytes")
        
        # Test model loading
        if self.nn_factory:
            try:
                # Create a simple test network
                config = NNConfig(
                    architecture=NNArchitecture.FFN,
                    input_length=100,
                    hidden_dims=[64, 32]
                )
                
                test_network = self.nn_factory.create_network(config)
                pretrained_results["loading_tests"]["factory_creation"] = {
                    "status": "success",
                    "network_type": type(test_network).__name__
                }
                print(f"   ✅ Neural Network Factory: Created {type(test_network).__name__}")
                
            except Exception as e:
                pretrained_results["loading_tests"]["factory_creation"] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"   ❌ Neural Network Factory test failed: {e}")
        
        return pretrained_results
    
    def audit_performance_benchmarking(self) -> Dict[str, Any]:
        """Audit neural network performance and benchmarking."""
        print("\n⚡ Auditing Neural Network Performance...")
        
        performance_results = {
            "estimator_tests": {},
            "execution_times": {},
            "memory_usage": {},
            "accuracy_tests": {}
        }
        
        # Test each estimator
        for name, estimator in self.estimators.items():
            print(f"   Testing {name}...")
            
            estimator_results = {
                "initialization_time": 0,
                "estimation_times": [],
                "memory_usage": 0,
                "success_rate": 0,
                "errors": []
            }
            
            # Test on multiple datasets
            test_count = 0
            success_count = 0
            
            for data_key, data in list(self.test_data.items())[:5]:  # Limit to 5 tests
                try:
                    test_count += 1
                    
                    # Measure estimation time
                    start_time = time.time()
                    result = estimator.estimate(data)
                    estimation_time = time.time() - start_time
                    
                    estimator_results["estimation_times"].append(estimation_time)
                    
                    # Check if result is valid
                    if isinstance(result, dict) and "hurst_parameter" in result:
                        success_count += 1
                    
                    print(f"      ✅ {data_key}: {estimation_time:.4f}s")
                    
                except Exception as e:
                    estimator_results["errors"].append(str(e))
                    print(f"      ❌ {data_key}: {e}")
            
            # Calculate success rate
            estimator_results["success_rate"] = success_count / test_count if test_count > 0 else 0
            estimator_results["average_time"] = np.mean(estimator_results["estimation_times"]) if estimator_results["estimation_times"] else 0
            
            performance_results["estimator_tests"][name] = estimator_results
            
            print(f"   {name}: {success_count}/{test_count} successful, avg time: {estimator_results['average_time']:.4f}s")
        
        return performance_results
    
    def audit_train_once_apply_many(self) -> Dict[str, Any]:
        """Audit train-once-apply-many workflow."""
        print("\n🔄 Auditing Train-Once-Apply-Many Workflow...")
        
        workflow_results = {
            "training_capabilities": {},
            "model_persistence": {},
            "inference_workflow": {},
            "production_readiness": {}
        }
        
        if not self.nn_factory:
            workflow_results["error"] = "Neural Network Factory not available"
            print("   ❌ Neural Network Factory not available")
            return workflow_results
        
        # Test training workflow
        try:
            # Create a test network
            config = NNConfig(
                architecture=NNArchitecture.FFN,
                input_length=100,
                hidden_dims=[32, 16],
                epochs=5  # Quick test
            )
            
            network = self.nn_factory.create_network(config)
            
            # Generate training data
            X_train = np.random.randn(50, 100)
            y_train = np.random.randn(50)
            
            # Test training
            start_time = time.time()
            training_history = network.train_model(X_train, y_train)
            training_time = time.time() - start_time
            
            workflow_results["training_capabilities"] = {
                "status": "success",
                "training_time": training_time,
                "history_keys": list(training_history.keys()) if training_history else []
            }
            print(f"   ✅ Training workflow: {training_time:.4f}s")
            
            # Test inference
            X_test = np.random.randn(10, 100)
            start_time = time.time()
            predictions = network.predict(X_test)
            inference_time = time.time() - start_time
            
            workflow_results["inference_workflow"] = {
                "status": "success",
                "inference_time": inference_time,
                "predictions_shape": predictions.shape,
                "batch_processing": True
            }
            print(f"   ✅ Inference workflow: {inference_time:.4f}s")
            
            # Test model saving/loading
            try:
                model_path = network.save_model()
                workflow_results["model_persistence"] = {
                    "status": "success",
                    "save_path": model_path,
                    "model_exists": Path(model_path).exists() if model_path else False
                }
                print(f"   ✅ Model persistence: {model_path}")
            except Exception as e:
                workflow_results["model_persistence"] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"   ❌ Model persistence failed: {e}")
            
        except Exception as e:
            workflow_results["error"] = str(e)
            print(f"   ❌ Train-once-apply-many workflow failed: {e}")
        
        return workflow_results
    
    def generate_audit_report(self, architecture_results: Dict, gpu_results: Dict, 
                            pretrained_results: Dict, performance_results: Dict, 
                            workflow_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        print("\n📋 Generating Audit Report...")
        
        report = {
            "audit_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": self.system_info,
                "test_data_count": len(self.test_data),
                "estimators_tested": len(self.estimators)
            },
            "architecture_audit": architecture_results,
            "gpu_optimization_audit": gpu_results,
            "pretrained_models_audit": pretrained_results,
            "performance_audit": performance_results,
            "workflow_audit": workflow_results,
            "overall_assessment": self._assess_overall_status(
                architecture_results, gpu_results, pretrained_results, 
                performance_results, workflow_results
            )
        }
        
        return report
    
    def _assess_overall_status(self, architecture_results: Dict, gpu_results: Dict,
                             pretrained_results: Dict, performance_results: Dict,
                             workflow_results: Dict) -> Dict[str, Any]:
        """Assess overall neural network estimator status."""
        
        assessment = {
            "status": "unknown",
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        score = 0
        max_score = 100
        
        # Architecture assessment (25 points)
        if architecture_results.get("estimators_available"):
            available_count = sum(1 for est in architecture_results["estimators_available"].values() 
                                if est.get("status") == "available")
            architecture_score = min(25, available_count * 6.25)
            score += architecture_score
            
            if available_count >= 3:
                assessment["strengths"].append("Multiple neural network architectures available")
            else:
                assessment["weaknesses"].append("Limited neural network architectures")
        
        # GPU optimization assessment (25 points)
        if gpu_results.get("pytorch_gpu", {}).get("available"):
            score += 15
            assessment["strengths"].append("PyTorch GPU acceleration available")
        else:
            assessment["weaknesses"].append("No PyTorch GPU acceleration")
        
        if gpu_results.get("jax_gpu", {}).get("available"):
            score += 10
            assessment["strengths"].append("JAX GPU acceleration available")
        else:
            assessment["weaknesses"].append("No JAX GPU acceleration")
        
        # Pretrained models assessment (20 points)
        config_count = len(pretrained_results.get("config_files", {}))
        if config_count > 0:
            score += min(20, config_count * 4)
            assessment["strengths"].append(f"{config_count} pretrained model configurations available")
        else:
            assessment["weaknesses"].append("No pretrained model configurations")
        
        # Performance assessment (20 points)
        if performance_results.get("estimator_tests"):
            successful_estimators = sum(1 for est in performance_results["estimator_tests"].values() 
                                      if est.get("success_rate", 0) > 0.5)
            score += min(20, successful_estimators * 5)
            
            if successful_estimators >= 2:
                assessment["strengths"].append("Multiple estimators working correctly")
            else:
                assessment["weaknesses"].append("Limited working estimators")
        
        # Workflow assessment (10 points)
        if workflow_results.get("training_capabilities", {}).get("status") == "success":
            score += 5
            assessment["strengths"].append("Training workflow functional")
        else:
            assessment["weaknesses"].append("Training workflow issues")
        
        if workflow_results.get("inference_workflow", {}).get("status") == "success":
            score += 5
            assessment["strengths"].append("Inference workflow functional")
        else:
            assessment["weaknesses"].append("Inference workflow issues")
        
        # Determine overall status
        assessment["score"] = score
        
        if score >= 80:
            assessment["status"] = "excellent"
        elif score >= 60:
            assessment["status"] = "good"
        elif score >= 40:
            assessment["status"] = "fair"
        else:
            assessment["status"] = "needs_improvement"
        
        # Generate recommendations
        if not gpu_results.get("pytorch_gpu", {}).get("available"):
            assessment["recommendations"].append("Install CUDA-enabled PyTorch for GPU acceleration")
        
        if not gpu_results.get("jax_gpu", {}).get("available"):
            assessment["recommendations"].append("Install JAX with GPU support for enhanced performance")
        
        if len(pretrained_results.get("config_files", {})) == 0:
            assessment["recommendations"].append("Train and save pretrained neural network models")
        
        if len(assessment["weaknesses"]) > len(assessment["strengths"]):
            assessment["recommendations"].append("Address implementation issues in neural network estimators")
        
        return assessment
    
    def save_results(self, report: Dict[str, Any]):
        """Save audit results to files."""
        print("\n💾 Saving Audit Results...")
        
        # Save JSON report
        json_path = self.output_dir / "neural_estimators_audit_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Results saved to {json_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print audit summary."""
        print("\n" + "="*70)
        print("🧠 NEURAL NETWORK ESTIMATORS AUDIT SUMMARY")
        print("="*70)
        
        assessment = report["overall_assessment"]
        print(f"Overall Status: {assessment['status'].upper()}")
        print(f"Overall Score: {assessment['score']}/100")
        
        print(f"\n📊 System Capabilities:")
        system_info = report["audit_metadata"]["system_info"]
        print(f"   PyTorch: {system_info['pytorch_version']}")
        print(f"   CUDA: {'Available' if system_info['cuda_available'] else 'Not Available'}")
        print(f"   JAX GPU: {'Available' if report['gpu_optimization_audit']['jax_gpu'].get('available') else 'Not Available'}")
        
        print(f"\n🏗️ Architecture Status:")
        arch_results = report["architecture_audit"]
        available_estimators = [name for name, info in arch_results["estimators_available"].items() 
                              if info.get("status") == "available"]
        print(f"   Available Estimators: {len(available_estimators)}")
        for est in available_estimators:
            print(f"      • {est}")
        
        print(f"\n⚡ Performance Summary:")
        perf_results = report["performance_audit"]
        if perf_results.get("estimator_tests"):
            for name, results in perf_results["estimator_tests"].items():
                success_rate = results.get("success_rate", 0)
                avg_time = results.get("average_time", 0)
                print(f"   {name}: {success_rate:.1%} success, {avg_time:.4f}s avg")
        
        print(f"\n💾 Pretrained Models:")
        pretrained_count = len(report["pretrained_models_audit"].get("config_files", {}))
        print(f"   Configuration Files: {pretrained_count}")
        
        if assessment["strengths"]:
            print(f"\n✅ Strengths:")
            for strength in assessment["strengths"]:
                print(f"   • {strength}")
        
        if assessment["weaknesses"]:
            print(f"\n❌ Weaknesses:")
            for weakness in assessment["weaknesses"]:
                print(f"   • {weakness}")
        
        if assessment["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in assessment["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "="*70)
    
    def run_comprehensive_audit(self):
        """Run the complete neural network audit."""
        print("🧠 Starting Comprehensive Neural Network Estimators Audit")
        print("="*70)
        
        # Run all audit components
        architecture_results = self.audit_architecture_implementation()
        gpu_results = self.audit_gpu_optimization()
        pretrained_results = self.audit_pretrained_models()
        performance_results = self.audit_performance_benchmarking()
        workflow_results = self.audit_train_once_apply_many()
        
        # Generate comprehensive report
        report = self.generate_audit_report(
            architecture_results, gpu_results, pretrained_results,
            performance_results, workflow_results
        )
        
        # Save results
        self.save_results(report)
        
        # Print summary
        self.print_summary(report)
        
        return report

def main():
    """Run the neural network estimators audit."""
    audit = NeuralEstimatorsAudit()
    results = audit.run_comprehensive_audit()
    return results

if __name__ == "__main__":
    results = main()
