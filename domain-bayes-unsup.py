import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import ast
import re
import argparse
import os
from datetime import datetime
from scipy.stats import entropy as shannon_entropy
from collections import Counter

# -------------------------------
# CLI argument for output dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="Output folder to store results")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def parse_list_safely(x):
    if isinstance(x, list):
        return [float(i) for i in x if is_number(i)]
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return [float(i) for i in parsed if is_number(i)]
    except Exception:
        pass
    # fallback to regex only if valid numbers exist
    return [float(i) for i in re.findall(r"\d+(?:\.\d+)?", str(x))]

def safe_entropy(seq):
    if len(seq) == 0:
        return 0
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return shannon_entropy(probs, base=2)

def safe_range(x):
    return max(x) - min(x) if len(x) > 1 else 0

def iqr(x):
    if len(x) < 2:
        return 0
    q75, q25 = np.percentile(x, [75 ,25])
    return q75 - q25

# -------------------------------
# Load and preprocess data
# -------------------------------
data = pd.read_csv("output_data.csv")

# -------------------------------
# Handle optional features
# -------------------------------

# If 'is_in_TI' is missing, add it and set to 0
if "is_in_TI" not in data.columns:
    print("⚠️ 'is_in_TI' not found in data — filling with 0")
    data["is_in_TI"] = 0
    data["is_in_tranco"] = 0

# Parse list fields safely before any derived operations
data["ttls"] = data["ttls"].apply(parse_list_safely)
data["answer_ips"] = data["answer_ips"].apply(parse_list_safely)

# Parse and clean
data["is_in_TI"] = data["is_in_TI"].fillna(0)
data["is_in_tranco"] = data["is_in_tranco"].fillna(0)
data["ttls"] = data["ttls"].apply(parse_list_safely)
data["answer_ips"] = data["answer_ips"].apply(parse_list_safely)

# Derived features
data["ttl_range"] = data["ttls"].apply(safe_range)
data["ttl_entropy"] = data["ttls"].apply(safe_entropy)
data["ttl_iqr"] = data["ttls"].apply(iqr)
data["ips_entropy"] = data["answer_ips"].apply(safe_entropy)
data["ips_count"] = data["answer_ips"].apply(lambda x: len(set(x)))

# -------------------------------
# Feature list including clustering flags
# -------------------------------
features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco",
    "ttl_range", "ttl_entropy", "ttl_iqr", "ips_entropy", "ips_count",
    "in_gmm_cluster", "in_iso_forest_cluster", "in_dbscan_cluster"
]

# Check for missing new features
for f in ["in_gmm_cluster", "in_iso_forest_cluster", "in_dbscan_cluster"]:
    if f not in data.columns:
        raise ValueError(f"Missing required column: {f}")

fft_features = [
    "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy"
]

data[fft_features] = data[fft_features].fillna(0)

X = data[features].values
N = len(data)

# Normalize features safely
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

import numpy as np

nan_rows = np.isnan(X_norm).any(axis=1)
nan_cols = np.isnan(X_norm).any(axis=0)

print("🧪 Rows with NaNs:")
print(data[nan_rows][features])


print("\n🧪 Columns with NaNs:")
for i, is_nan in enumerate(nan_cols):
    if is_nan:
        print(f" - {features[i]}")

# -------------------------------
# Bayesian model
# -------------------------------
with pm.Model() as model:
    theta = pm.Beta("theta", alpha=1, beta=1)

    malicious = pm.Bernoulli("malicious", p=theta, shape=N)

    mu_0 = pm.Normal("mu_0", mu=0, sigma=1, shape=X.shape[1])
    mu_1 = pm.Normal("mu_1", mu=0, sigma=1, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=1, shape=X.shape[1])

    mu = mu_0 * (1 - malicious[:, None]) + mu_1 * malicious[:, None]

    X_obs = pm.Normal("X_obs", mu=mu, sigma=sigma, observed=X_norm)

    if np.isnan(X_norm).any():
        raise ValueError("❌ X_norm contains NaNs! Check your preprocessing and feature computation.")

    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,  # reduce for debugging
        #chains=4,  # reduce for debugging
        #cores=2,
        cores=4,
        init="adapt_diag",
        target_accept=0.99,
        nuts={"max_treedepth": 15},
        progressbar=True
    )


# -------------------------------
# Posterior analysis
# -------------------------------
posterior_m = trace.posterior["malicious"].mean(dim=["chain", "draw"]).values
data["p_malicious"] = posterior_m

# -------------------------------
# Plot: Posterior P(malicious)
# -------------------------------
plt.figure(figsize=(12, 6))
sorted_data = data.sort_values("p_malicious", ascending=False)
plt.barh(sorted_data["query"], sorted_data["p_malicious"], color="tomato")
plt.xlabel("P(domain is malicious)")
plt.title("Posterior Probability of Maliciousness per Domain")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig(os.path.join(args.output_dir, "posterior_maliciousness.png"))
plt.close()

# -------------------------------
# Plot: Feature means by class
# -------------------------------
az.plot_forest(trace, var_names=["mu_0", "mu_1"], combined=True)
plt.title("Feature Means by Class (Benign vs Malicious)")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "feature_means_forest.png"))
plt.close()

# -------------------------------
# Trace diagnostics
# -------------------------------
az.plot_trace(trace, var_names=["theta", "mu_0", "mu_1"])
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "trace_diagnostics.png"))
plt.close()

# -------------------------------
# Summary + log
# -------------------------------
output_file = os.path.join(args.output_dir, "maliciousness_report.csv")
final_report = data[["query", "p_malicious"] + features].sort_values("p_malicious", ascending=False)
final_report.to_csv(output_file, index=False)

log_file = os.path.join(args.output_dir, "run_log.md")
with open(log_file, "w") as f:
    f.write(f"# Run Log: Bayesian DNS Analysis\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Domains analyzed: {len(data)}\n")
    f.write(f"Features used: {', '.join(features)}\n")
    f.write(f"Output files:\n")
    f.write(f"- maliciousness_report.csv\n")
    f.write(f"- posterior_maliciousness.png\n")
    f.write(f"- feature_means_forest.png\n")
    f.write(f"- trace_diagnostics.png\n")

# -------------------------------
# Print to console
# -------------------------------
print("\n🔍 Final Domain Maliciousness Probabilities:\n")
for _, row in final_report.iterrows():
    print(f"{row['query']:<25}  P(malicious) = {row['p_malicious']:.3f}")

print(f"\n✅ Outputs saved in: {args.output_dir}")

