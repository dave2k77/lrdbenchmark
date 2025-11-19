
import numpy as np
import matplotlib.pyplot as plt
from lrdbenchmark.models.data_models.fgn.fgn_model import FractionalGaussianNoise
from lrdbenchmark.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator

def verify_fgn_variance():
    print("Verifying fGn Variance...")
    H = 0.7
    sigma = 1.0
    N = 10000
    fgn = FractionalGaussianNoise(H=H, sigma=sigma)
    data = fgn.generate(N, random_state=42)
    
    print(f"Generated fGn length: {len(data)}")
    print(f"Mean: {np.mean(data):.4f} (Expected ~0)")
    print(f"Std: {np.std(data):.4f} (Expected ~{sigma})")
    
    # Check autocorrelation at lag 1
    # rho(1) = 0.5 * (2^(2H) - 2)
    expected_rho1 = 0.5 * (2**(2*H) - 2)
    
    # Calculate sample autocorrelation
    acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    acf = acf[len(acf)//2:]
    acf /= acf[0]
    rho1 = acf[1]
    
    print(f"Lag-1 Autocorrelation: {rho1:.4f} (Expected: {expected_rho1:.4f})")
    
    if abs(rho1 - expected_rho1) < 0.05:
        print("SUCCESS: fGn correlation structure matches theory.")
    else:
        print("FAILURE: fGn correlation structure mismatch.")

def verify_whittle_estimator():
    print("\nVerifying Whittle Estimator...")
    estimator = WhittleEstimator()
    
    for H_true in [0.3, 0.5, 0.7, 0.9]:
        fgn = FractionalGaussianNoise(H=H_true, sigma=1.0)
        # Generate longer series for better estimation
        data = fgn.generate(4096, random_state=42)
        
        result = estimator.estimate(data)
        H_est = result["hurst_parameter"]
        
        print(f"True H: {H_true:.2f}, Estimated H: {H_est:.4f}, Error: {abs(H_est - H_true):.4f}")
        
        if abs(H_est - H_true) < 0.05:
            print(f"SUCCESS: Estimation within tolerance for H={H_true}")
        else:
            print(f"FAILURE: Estimation error too high for H={H_true}")

if __name__ == "__main__":
    try:
        verify_fgn_variance()
        verify_whittle_estimator()
    except Exception as e:
        print(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
