import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# -------------------------------
# Load and preprocess data
# -------------------------------
data = pd.read_csv("dns_summary.csv")

# Clean up missing boolean columns
data["is_in_TI"] = data["is_in_TI"].fillna(0)
data["is_in_tranco"] = data["is_in_tranco"].fillna(0)

# Features
features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco"
]

X = data[features].values
N = len(data)

# -------------------------------
# Bayesian model with PyMC
# -------------------------------
with pm.Model() as model:
    # Prior probability of malicious domain
    theta = pm.Beta("theta", alpha=1, beta=1)

    # Latent true label for each domain (0 = benign, 1 = malicious)
    malicious = pm.Bernoulli("malicious", p=theta, shape=N)

    # Define class-dependent feature distributions
    mu_0 = pm.Normal("mu_0", mu=0, sigma=2, shape=len(features))  # benign means
    mu_1 = pm.Normal("mu_1", mu=0, sigma=2, shape=len(features))  # malicious means
    sigma = pm.HalfNormal("sigma", sigma=1, shape=len(features))  # shared stddev

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    # Expected means based on latent class
    mu = mu_0 * (1 - malicious[:, None]) + mu_1 * malicious[:, None]

    # Likelihood for each feature (vectorized)
    observed = pm.Normal("X", mu=mu, sigma=sigma, observed=X_norm)

    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95)

# -------------------------------
# Extract posterior maliciousness
# -------------------------------
posterior_m = trace.posterior["malicious"].mean(dim=["chain", "draw"]).values
data["p_malicious"] = posterior_m

# -------------------------------
# Plot posterior maliciousness
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
# Interpretability: feature means
# -------------------------------
az.plot_forest(trace, var_names=["mu_0", "mu_1"], combined=True)
plt.title("Feature Means by Class (Benign vs Malicious)")
plt.show()
