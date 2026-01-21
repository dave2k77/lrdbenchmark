#!/usr/bin/env python3
"""
Real-World Data Validation for LRDBenchmark (Simplified Version)

This script downloads and processes real-world datasets from various sources
to validate LRD estimation performance on actual data.

Data Sources:
- Financial: Yahoo Finance (S&P 500, Bitcoin, Gold)
- Physiological: Simulated HRV and EEG data
- Climate: Simulated temperature and precipitation data
- Network: Simulated internet traffic data
- Biophysics: Simulated protein folding data

Author: LRDBench Development Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import LRDBenchmark components
from lrdbenchmark.analysis.temporal.rs.rs_estimator_unified import RSEstimator
from lrdbenchmark.analysis.temporal.dfa.dfa_estimator_unified import DFAEstimator
from lrdbenchmark.analysis.spectral.whittle.whittle_estimator_unified import WhittleEstimator

# Data download libraries
import yfinance as yf

class RealWorldDataValidator:
    """Real-world data validation for LRD estimation"""
    
    def __init__(self, results_dir="results/real_world_validation"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize estimators (using only classical for simplicity)
        self.estimators = {
            'R/S': RSEstimator(),
            'DFA': DFAEstimator(),
            'Whittle': WhittleEstimator()
        }
        
        self.results = []
        
    def download_financial_data(self):
        """Download financial time series data from Yahoo Finance"""
        print("📈 Downloading financial data...")
        
        financial_data = {}
        
        # Define financial instruments
        symbols = {
            'SP500': '^GSPC',  # S&P 500
            'Bitcoin': 'BTC-USD',  # Bitcoin
            'Gold': 'GC=F',  # Gold futures
            'VIX': '^VIX',  # Volatility index
        }
        
        for name, symbol in symbols.items():
            try:
                print(f"  Downloading {name} ({symbol})...")
                ticker = yf.Ticker(symbol)
                
                # Download 3 years of daily data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=3*365)
                
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Use adjusted close prices and calculate returns
                    prices = data['Close'].values
                    returns = np.diff(np.log(prices))  # Log returns
                    
                    # Remove NaN values
                    returns = returns[~np.isnan(returns)]
                    
                    if len(returns) > 500:  # Ensure sufficient data
                        financial_data[name] = {
                            'data': returns,
                            'length': len(returns),
                            'source': f'Yahoo Finance ({symbol})',
                            'description': f'{name} daily log returns',
                            'domain': 'Financial'
                        }
                        print(f"    ✅ {name}: {len(returns)} data points")
                    else:
                        print(f"    ⚠️ {name}: Insufficient data ({len(returns)} points)")
                else:
                    print(f"    ❌ {name}: No data available")
                    
            except Exception as e:
                print(f"    ❌ {name}: Error - {str(e)}")
                
        return financial_data
    
    def generate_simulated_data(self):
        """Generate simulated data for other domains"""
        print("🧬 Generating simulated data for other domains...")
        
        simulated_data = {}
        np.random.seed(42)
        
        # Heart Rate Variability (HRV)
        print("  Generating Heart Rate Variability data...")
        hrv_data = self._generate_fgn_like_data(1500, H=0.7, sigma=1.0)
        simulated_data['HRV'] = {
            'data': hrv_data,
            'length': len(hrv_data),
            'source': 'PhysioNet (simulated)',
            'description': 'Heart Rate Variability (RR intervals)',
            'domain': 'Physiological'
        }
        
        # EEG data
        print("  Generating EEG data...")
        eeg_data = self._generate_fgn_like_data(1200, H=0.6, sigma=0.5)
        simulated_data['EEG'] = {
            'data': eeg_data,
            'length': len(eeg_data),
            'source': 'PhysioNet (simulated)',
            'description': 'EEG alpha rhythm',
            'domain': 'Physiological'
        }
        
        # Temperature data
        print("  Generating temperature data...")
        temp_data = self._generate_fgn_like_data(1000, H=0.7, sigma=0.3)
        simulated_data['Temperature'] = {
            'data': temp_data,
            'length': len(temp_data),
            'source': 'NOAA (simulated)',
            'description': 'Global temperature anomalies',
            'domain': 'Climate'
        }
        
        # Network traffic
        print("  Generating network traffic data...")
        traffic_data = self._generate_fgn_like_data(2000, H=0.8, sigma=0.2)
        simulated_data['Internet_Traffic'] = {
            'data': traffic_data,
            'length': len(traffic_data),
            'source': 'CAIDA (simulated)',
            'description': 'Internet backbone traffic',
            'domain': 'Network'
        }
        
        # Protein folding
        print("  Generating protein folding data...")
        protein_data = self._generate_fgn_like_data(800, H=0.7, sigma=0.3)
        simulated_data['Protein_Folding'] = {
            'data': protein_data,
            'length': len(protein_data),
            'source': 'PDB (simulated)',
            'description': 'Protein folding trajectory (RMSD)',
            'domain': 'Biophysics'
        }
        
        print("    ✅ Simulated data generated")
        return simulated_data
    
    def _generate_fgn_like_data(self, n, H, sigma=1.0):
        """Generate FGN-like data with specified Hurst parameter"""
        # Simple FGN generation using fractional differencing
        white_noise = np.random.normal(0, sigma, n)
        
        # Apply fractional differencing to create LRD
        d = H - 0.5  # Fractional differencing parameter
        
        if d > 0:
            # Apply moving average with weights
            import math
            weights = np.array([1.0] + [d * (1-d)**(k-1) / math.factorial(k) for k in range(1, min(50, n))])
            weights = weights / np.sum(weights)
            
            # Apply convolution
            fgn_data = np.convolve(white_noise, weights, mode='same')
        else:
            fgn_data = white_noise
            
        return fgn_data
    
    def estimate_lrd(self, data, dataset_name):
        """Estimate LRD for a given dataset using all estimators"""
        print(f"  Estimating LRD for {dataset_name}...")
        
        dataset_results = {
            'dataset': dataset_name,
            'length': len(data),
            'estimates': {}
        }
        
        for estimator_name, estimator in self.estimators.items():
            try:
                # Estimate Hurst parameter
                result = estimator.estimate(data)
                
                if result is not None and 'hurst_parameter' in result:
                    hurst_est = result['hurst_parameter']
                    dataset_results['estimates'][estimator_name] = {
                        'hurst_parameter': hurst_est,
                        'success': True,
                        'error': None
                    }
                    print(f"    ✅ {estimator_name}: H = {hurst_est:.3f}")
                else:
                    dataset_results['estimates'][estimator_name] = {
                        'hurst_parameter': None,
                        'success': False,
                        'error': 'Estimation failed'
                    }
                    print(f"    ❌ {estimator_name}: Estimation failed")
                    
            except Exception as e:
                dataset_results['estimates'][estimator_name] = {
                    'hurst_parameter': None,
                    'success': False,
                    'error': str(e)
                }
                print(f"    ❌ {estimator_name}: Error - {str(e)}")
        
        return dataset_results
    
    def run_validation(self):
        """Run complete real-world data validation"""
        print("🚀 Starting Real-World Data Validation")
        print("=" * 50)
        
        # Download financial data
        financial_data = self.download_financial_data()
        
        # Generate simulated data
        simulated_data = self.generate_simulated_data()
        
        # Combine all datasets
        all_datasets = {**financial_data, **simulated_data}
        
        print(f"\n📊 Total datasets: {len(all_datasets)}")
        print("=" * 50)
        
        # Estimate LRD for each dataset
        for dataset_name, dataset_info in all_datasets.items():
            print(f"\n🔍 Analyzing {dataset_name} ({dataset_info['domain']})")
            print(f"  Source: {dataset_info['source']}")
            print(f"  Description: {dataset_info['description']}")
            print(f"  Length: {dataset_info['length']} points")
            
            # Estimate LRD
            results = self.estimate_lrd(dataset_info['data'], dataset_name)
            results['domain'] = dataset_info['domain']
            results['source'] = dataset_info['source']
            results['description'] = dataset_info['description']
            
            self.results.append(results)
        
        # Generate summary and visualizations
        self.generate_summary()
        self.create_visualizations()
        
        print("\n✅ Real-World Data Validation Complete!")
        print(f"Results saved to: {self.results_dir}")
    
    def generate_summary(self):
        """Generate summary statistics and tables"""
        print("\n📊 Generating Summary Statistics...")
        
        # Create summary table
        summary_data = []
        
        for result in self.results:
            dataset_name = result['dataset']
            domain = result['domain']
            length = result['length']
            
            # Count successful estimations
            successful_estimates = [est for est in result['estimates'].values() if est['success']]
            success_rate = len(successful_estimates) / len(result['estimates'])
            
            # Calculate average Hurst parameter
            hurst_values = [est['hurst_parameter'] for est in successful_estimates if est['hurst_parameter'] is not None]
            avg_hurst = np.mean(hurst_values) if hurst_values else None
            std_hurst = np.std(hurst_values) if len(hurst_values) > 1 else 0
            
            summary_data.append({
                'Dataset': dataset_name,
                'Domain': domain,
                'Length': length,
                'Success_Rate': success_rate,
                'Avg_Hurst': avg_hurst,
                'Std_Hurst': std_hurst,
                'Successful_Estimators': len(hurst_values)
            })
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.results_dir, 'real_world_validation_summary.csv'), index=False)
        
        # Print summary
        print("\n📋 Real-World Data Validation Summary:")
        print("=" * 80)
        print(f"{'Dataset':<20} {'Domain':<12} {'Length':<8} {'Success':<8} {'Avg H':<8} {'Std H':<8}")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Dataset']:<20} {row['Domain']:<12} {row['Length']:<8} {row['Success_Rate']:<8.2f} {row['Avg_Hurst']:<8.3f} {row['Std_Hurst']:<8.3f}")
        
        # Overall statistics
        total_datasets = len(summary_df)
        avg_success_rate = summary_df['Success_Rate'].mean()
        total_estimations = len(summary_df) * len(self.estimators)
        successful_estimations = (summary_df['Success_Rate'] * len(self.estimators)).sum()
        
        print("\n📈 Overall Statistics:")
        print(f"  Total Datasets: {total_datasets}")
        print(f"  Average Success Rate: {avg_success_rate:.2%}")
        print(f"  Total Estimations: {total_estimations}")
        print(f"  Successful Estimations: {successful_estimations:.0f}")
        print(f"  Overall Success Rate: {successful_estimations/total_estimations:.2%}")
        
        # Save detailed results
        import json
        with open(os.path.join(self.results_dir, 'real_world_validation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def create_visualizations(self):
        """Create visualizations of real-world validation results"""
        print("\n📊 Creating Visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Success Rate by Domain
        domain_success = {}
        for result in self.results:
            domain = result['domain']
            if domain not in domain_success:
                domain_success[domain] = []
            
            successful = sum(1 for est in result['estimates'].values() if est['success'])
            total = len(result['estimates'])
            domain_success[domain].append(successful / total)
        
        domain_avg_success = {domain: np.mean(rates) for domain, rates in domain_success.items()}
        
        axes[0, 0].bar(domain_avg_success.keys(), domain_avg_success.values())
        axes[0, 0].set_title('Success Rate by Domain')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Hurst Parameter Distribution
        all_hurst_values = []
        for result in self.results:
            for est in result['estimates'].values():
                if est['success'] and est['hurst_parameter'] is not None:
                    all_hurst_values.append(est['hurst_parameter'])
        
        axes[0, 1].hist(all_hurst_values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Estimated Hurst Parameters')
        axes[0, 1].set_xlabel('Hurst Parameter')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0.5, color='red', linestyle='--', label='H = 0.5 (No LRD)')
        axes[0, 1].legend()
        
        # 3. Estimator Performance
        estimator_success = {}
        for estimator_name in self.estimators.keys():
            success_count = 0
            total_count = 0
            for result in self.results:
                if estimator_name in result['estimates']:
                    total_count += 1
                    if result['estimates'][estimator_name]['success']:
                        success_count += 1
            estimator_success[estimator_name] = success_count / total_count if total_count > 0 else 0
        
        axes[1, 0].bar(estimator_success.keys(), estimator_success.values())
        axes[1, 0].set_title('Estimator Success Rate')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Dataset Length vs Success Rate
        dataset_lengths = [result['length'] for result in self.results]
        dataset_success_rates = []
        for result in self.results:
            successful = sum(1 for est in result['estimates'].values() if est['success'])
            total = len(result['estimates'])
            dataset_success_rates.append(successful / total)
        
        axes[1, 1].scatter(dataset_lengths, dataset_success_rates, alpha=0.7)
        axes[1, 1].set_title('Dataset Length vs Success Rate')
        axes[1, 1].set_xlabel('Dataset Length')
        axes[1, 1].set_ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'real_world_validation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("    ✅ Visualizations saved")


if __name__ == "__main__":
    # Run real-world validation
    validator = RealWorldDataValidator()
    validator.run_validation()
