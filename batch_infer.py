import pandas as pd
import numpy as np
import arviz as az
from scipy.stats import norm

# Load trace and normalization stats
trace = az.from_netcdf("trace.nc")
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")

# Same features as training
features = [...]  # use your list here

# Load new CSV (no labels needed)
df = pd.read_csv("new_domains.csv")
X = df[features].values
X_norm = (X - X_mean) / X_std

# Posterior samples
mu_0 = trace.posterior["mu_0"].stack(samples=("chain", "draw")).values
mu_1 = trace.posterior["mu_1"].stack(samples=("chain", "draw")).values
sigma = trace.posterior["sigma"].stack(samples=("chain", "draw")).values
theta = trace.posterior["theta"].stack(samples=("chain", "draw")).values.squeeze()

# Predict for all rows
probs = []
for x in X_norm:
    log_p_benign = norm.logpdf(x, mu_0, sigma).sum(axis=1) + np.log(1 - theta)
    log_p_malicious = norm.logpdf(x, mu_1, sigma).sum(axis=1) + np.log(theta)
    p = 1 / (1 + np.exp(log_p_benign - log_p_malicious))
    probs.append(np.mean(p))

df["p_malicious"] = probs
df.to_csv("batch_inference_output.csv", index=False)
print("✅ Predictions saved to batch_inference_output.csv")

