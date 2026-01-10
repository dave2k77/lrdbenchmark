import pandas as pd
import json
import sys

base_dir = "benchmark_results_ml_nn_v2"

def print_md(df):
    try:
        print(df.to_markdown(floatfmt=".3f"))
    except ImportError:
        print(df.to_string(float_format=lambda x: "{:.3f}".format(x)))

print("# ML and NN Estimator Benchmark Results\n")

# Leaderboard
try:
    lb = pd.read_csv(f"{base_dir}/leaderboard.csv")
    print("## Leaderboard\n")
    print_md(lb)
    
    # Save as separate md file for artifacts
    with open(f"{base_dir}/leaderboard.md", "w") as f:
        try:
            f.write(lb.to_markdown(index=False, floatfmt=".4f"))
        except:
            f.write(lb.to_string())
            
except Exception as e:
    print(f"Error reading leaderboard: {e}")

# Pure Stats (Table 1)
print("\n## Table 1: Performance on Pure Synthetic Data (RMSE)\n")
try:
    pure = pd.read_csv(f"{base_dir}/pure_results.csv")
    # Pivot: Index=[Model, True_H], Columns=Estimator
    pivot = pure.groupby(["Model", "True_H", "Estimator"])["SqError"].mean().apply(lambda x: x**0.5).unstack()
    print_md(pivot)
    
    with open(f"{base_dir}/table_pure_rmse.md", "w") as f:
        try:
            f.write(pivot.to_markdown(floatfmt=".3f"))
        except:
            f.write(pivot.to_string())

except Exception as e:
    print(f"Error creating Table 1: {e}")

# Robustness Stats (Table 2)
print("\n## Table 2: Robustness to Contamination (MAE)\n")
try:
    cont = pd.read_csv(f"{base_dir}/contaminated_results.csv")
    # Pivot: Index=Contamination, Columns=Estimator
    pivot_cont = cont.pivot_table(index="Contamination", columns="Estimator", values="AbsError", aggfunc="mean")
    print_md(pivot_cont)
    
    with open(f"{base_dir}/table_robustness_mae.md", "w") as f:
        try:
            f.write(pivot_cont.to_markdown(floatfmt=".3f"))
        except:
            f.write(pivot_cont.to_string())

except Exception as e:
    print(f"Error creating Table 2: {e}")
