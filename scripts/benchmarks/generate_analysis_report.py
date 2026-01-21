import pandas as pd
import numpy as np
from pathlib import Path

def generate_markdown_report(results_dir: str = "benchmark_results_full"):
    results_path = Path(results_dir)
    
    # Load results
    try:
        df_pure = pd.read_csv(results_path / "pure_results.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_pure = pd.DataFrame()

    try:
        df_cont = pd.read_csv(results_path / "contaminated_results.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_cont = pd.DataFrame()

    try:
        df_real = pd.read_csv(results_path / "real_world_results.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_real = pd.DataFrame()

    report = []
    report.append("# Analytical Report: Benchmarking LRD Estimators\n")
    
    # --- Pure Data ---
    report.append("## 1. Pure Synthetic Processes\n")
    report.append("### 1.1 Accuracy (RMSE)\n")
    
    # Calculate RMSE for fGn at max length
    max_len = df_pure["Length"].max()
    fgn_data = df_pure[(df_pure["Model"] == "fGn") & (df_pure["Length"] == max_len)]
    
    if not fgn_data.empty:
        rmse = fgn_data.groupby("Estimator").apply(
            lambda x: np.sqrt((x["SqError"]).mean())
        ).sort_values()
        
        report.append(f"**Model: fGn, N={max_len}**\n")
        report.append("| Estimator | RMSE |")
        report.append("|---|---|")
        for est, val in rmse.items():
            report.append(f"| {est} | {val:.4f} |")
        report.append("\n")
        
        best_est = rmse.index[0]
        report.append(f"**Observation**: The **{best_est}** estimator achieved the lowest RMSE on pure fGn data.\n")

    # --- Contamination ---
    report.append("## 2. Robustness to Contamination\n")
    
    if not df_cont.empty:
        # Calculate mean absolute error at max contamination level
        max_level = df_cont["Level"].max()
        cont_data = df_cont[df_cont["Level"] == max_level]
        
        robustness = cont_data.groupby("Estimator")["AbsError"].mean().sort_values()
        
        report.append(f"### 2.1 Performance under Severe Contamination (Level {max_level})\n")
        report.append("| Estimator | Mean Abs Error |")
        report.append("|---|---|")
        for est, val in robustness.items():
            report.append(f"| {est} | {val:.4f} |")
        report.append("\n")
        
        most_robust = robustness.index[0]
        report.append(f"**Observation**: The **{most_robust}** estimator demonstrated the highest robustness.\n")

    # --- Real World ---
    report.append("## 3. Real-World Time Series\n")
    
    if not df_real.empty:
        report.append("### 3.1 Mean Hurst Estimates\n")
        
        real_summary = df_real.groupby(["Dataset", "Estimator"])["Estimate"].mean().unstack()
        
        # Manual markdown table
        header = "| Dataset | " + " | ".join(real_summary.columns) + " |"
        separator = "|---|" + "---|" * len(real_summary.columns)
        report.append(header)
        report.append(separator)
        
        for idx, row in real_summary.iterrows():
            row_str = " | ".join([f"{val:.3f}" for val in row])
            report.append(f"| {idx} | {row_str} |")
        report.append("\n")

    # Save report
    with open(results_path / "final_analysis_report.md", "w") as f:
        f.write("\n".join(report))
    
    print(f"Report generated at {results_path / 'final_analysis_report.md'}")

if __name__ == "__main__":
    generate_markdown_report()
