import pandas as pd
import numpy as np
import arviz as az
from scipy.stats import norm

# ----------------------------------------
# Feature list (24 total)
# ----------------------------------------
features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco",
    "ttl_range", "ttl_entropy", "ttl_iqr", "ips_entropy", "ips_count",
    "in_gmm_cluster", "in_isoforest_cluster", "in_dbscan_cluster"
]

# ----------------------------------------
# Load CSV of new domains
# ----------------------------------------
df = pd.read_csv("new_domains.csv")

# Safety check
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

X = df[features].values

# ----------------------------------------
# Load model artifacts
# ----------------------------------------
trace = az.from_netcdf("trace.nc")
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")
X_std[X_std == 0] = 1

X_norm = (X - X_mean) / X_std

# ----------------------------------------
# Extract posterior parameters
# ----------------------------------------
mu_0 = trace.posterior["mu_0"].stack(samples=("chain", "draw")).values.T
mu_1 = trace.posterior["mu_1"].stack(samples=("chain", "draw")).values.T
sigma = trace.posterior["sigma"].stack(samples=("chain", "draw")).values.T
theta = trace.posterior["theta"].stack(samples=("chain", "draw")).values.squeeze()

# ----------------------------------------
# Predict per row
# ----------------------------------------
probs = []
for x in X_norm:
    x_rep = np.repeat([x], mu_0.shape[0], axis=0)
    log_p_benign = norm.logpdf(x_rep, loc=mu_0, scale=sigma).sum(axis=1) + np.log(1 - theta)
    log_p_malicious = norm.logpdf(x_rep, loc=mu_1, scale=sigma).sum(axis=1) + np.log(theta)
    p = 1 / (1 + np.exp(log_p_benign - log_p_malicious))
    probs.append(np.mean(p))

df["p_malicious"] = probs

# ----------------------------------------
# Output results
# ----------------------------------------
df.to_csv("batch_inference_output.csv", index=False)
print("✅ Predictions saved to: batch_inference_output.csv")

