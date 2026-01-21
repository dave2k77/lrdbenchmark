
import numpy as np
import time
from lrdbenchmark.models.data_models import FBMModel, FGNModel
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.wavelet.cwt.cwt_estimator_unified import CWTEstimator

def test_estimator(name, estimator, data, true_hurst):
    try:
        start_time = time.time()
        result = estimator.estimate(data)
        elapsed = time.time() - start_time
        estimated_hurst = result.get("hurst_parameter", np.nan)
        error = abs(estimated_hurst - true_hurst)
        print(f"{name}: True H={true_hurst:.2f}, Est H={estimated_hurst:.4f}, Error={error:.4f}, Time={elapsed:.4f}s")
        return error
    except Exception as e:
        print(f"{name}: Failed with error: {e}")
        return np.nan

def main():
    print("🚀 Starting DFA and CWT Tuning Script")
    
    # Generate test data
    n_samples = 2000 # Increased sample size slightly for better stability
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    rng = np.random.default_rng(42)
    
    test_data = {}
    for H in hurst_values:
        fgn_model = FGNModel(H=H)
        test_data[H] = fgn_model.generate(length=n_samples, rng=rng)

    # Initialize estimators with default parameters (to reproduce issues)
    print("\n--- Baseline Performance (Defaults) ---")
    dfa = DFAEstimator()
    cwt = CWTEstimator()
    
    dfa_errors = []
    cwt_errors = []
    
    for H, data in test_data.items():
        dfa_errors.append(test_estimator("DFA", dfa, data, H))
        cwt_errors.append(test_estimator("CWT", cwt, data, H))
        
    print(f"\nMean DFA Error: {np.nanmean(dfa_errors):.4f}")
    print(f"Mean CWT Error: {np.nanmean(cwt_errors):.4f}")

if __name__ == "__main__":
    main()
