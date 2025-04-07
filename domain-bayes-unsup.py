import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import ast
import re
from scipy.stats import entropy as shannon_entropy
from collections import Counter

# -------------------------------
# Helper functions
# -------------------------------
def parse_list_safely(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return re.findall(r"[\d\.]+", str(x))

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
data = pd.read_csv("dns_summary.csv")

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

# Final feature list
features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco",
    "ttl_range", "ttl_entropy", "ttl_iqr", "ips_entropy", "ips_count"
]

X = data[features].values
N = len(data)

# Normalize features safely
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

# -------------------------------
# Bayesian model
# -------------------------------
with pm.Model() as model:
    theta = pm.Beta("theta", alpha=1, beta=1)

    # Latent label per domain
    malicious = pm.Bernoulli("malicious", p=theta, shape=N)

    # Feature priors (tightened for stability)
    mu_0 = pm.Normal("mu_0", mu=0, sigma=1, shape=X.shape[1])
    mu_1 = pm.Normal("mu_1", mu=0, sigma=1, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=1, shape=X.shape[1])

    # Mixture of class-based distributions
    mu = mu_0 * (1 - malicious[:, None]) + mu_1 * malicious[:, None]

    # Likelihood
    X_obs = pm.Normal("X_obs", mu=mu, sigma=sigma, observed=X_norm)

    # Improved sampling
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
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
plt.show()

# -------------------------------
# Plot: Feature means by class
# -------------------------------
az.plot_forest(trace, var_names=["mu_0", "mu_1"], combined=True)
plt.title("Feature Means by Class (Benign vs Malicious)")
plt.tight_layout()
plt.show()

# -------------------------------
# ArviZ diagnostics
# -------------------------------
summary = az.summary(trace, var_names=["theta", "mu_0", "mu_1", "sigma"])
print("\n📋 Diagnostic Summary:\n", summary)
az.plot_trace(trace, var_names=["theta", "mu_0", "mu_1"])
plt.tight_layout()
plt.show()

# -------------------------------
# Print + Log final results
# -------------------------------
output_file = "maliciousness_report.csv"
columns_to_show = ["query", "p_malicious"] + features

final_report = data[columns_to_show].sort_values("p_malicious", ascending=False)

# Print to console
print("\n🔍 Final Domain Maliciousness Probabilities:\n")
for _, row in final_report.iterrows():
    print(f"{row['query']:<25}  P(malicious) = {row['p_malicious']:.3f}")

# Save to CSV
final_report.to_csv(output_file, index=False)
print(f"\n✅ Full report saved to: {output_file}")

# Save artifacts after main training script
az.to_netcdf(trace, "trace.nc")
np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)
