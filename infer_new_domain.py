import argparse
import numpy as np
import arviz as az
from scipy.stats import norm

# ----------------------------------------
# Feature list (same order as in training)
# ----------------------------------------
features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco",
    "ttl_range", "ttl_entropy", "ttl_iqr", "ips_entropy", "ips_count"
]

# ----------------------------------------
# Parse CLI input
# ----------------------------------------
parser = argparse.ArgumentParser()
for feat in features:
    parser.add_argument(f"--{feat}", type=float, required=True)
args = parser.parse_args()

input_features = np.array([[getattr(args, feat) for feat in features]])

# ----------------------------------------
# Load trained artifacts
# ----------------------------------------
trace = az.from_netcdf("trace.nc")
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")

# ----------------------------------------
# Normalize the new input
# ----------------------------------------
X_std[X_std == 0] = 1  # Safety
input_norm = (input_features - X_mean) / X_std

# ----------------------------------------
# Extract posterior samples
# ----------------------------------------
mu_0 = trace.posterior["mu_0"].stack(samples=("chain", "draw")).values.T  # shape: (samples, features)
mu_1 = trace.posterior["mu_1"].stack(samples=("chain", "draw")).values.T
sigma = trace.posterior["sigma"].stack(samples=("chain", "draw")).values.T
theta = trace.posterior["theta"].stack(samples=("chain", "draw")).values.squeeze()

# ----------------------------------------
# Compute log-likelihoods per sample
# ----------------------------------------
x_repeated = np.repeat(input_norm, mu_0.shape[0], axis=0)

log_p_benign = norm.logpdf(x_repeated, loc=mu_0, scale=sigma).sum(axis=1) + np.log(1 - theta)
log_p_malicious = norm.logpdf(x_repeated, loc=mu_1, scale=sigma).sum(axis=1) + np.log(theta)

# ----------------------------------------
# Posterior probability via Bayes rule
# ----------------------------------------
posterior_probs = 1 / (1 + np.exp(log_p_benign - log_p_malicious))

# ----------------------------------------
# Output
# ----------------------------------------
print(f"\n🔍 Estimated P(malicious): {posterior_probs.mean():.3f}")
print(f"📏 95% credible interval:  ({np.percentile(posterior_probs, 2.5):.3f}, {np.percentile(posterior_probs, 97.5):.3f})")

