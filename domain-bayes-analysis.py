import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# -----------------------------
# 1. Sample input data (mock)
# -----------------------------
# You can replace this with your real feature data
data = pd.DataFrame({
    "ttl": [60, 300, 100, 1800, 50],
    "num_clients": [2, 10, 1, 20, 1],
    "entropy": [0.9, 0.5, 0.95, 0.4, 0.97],
    "ti_flag": [1, 0, None, 0, 1]  # External labels: 1 = malicious, 0 = benign
})

# Keep only rows with external evidence
observed_data = data.dropna(subset=["ti_flag"])
X = observed_data[["ttl", "num_clients", "entropy"]].values
y = observed_data["ti_flag"].values.astype(int)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# -----------------------------
# 2. Build Bayesian Model
# -----------------------------
with pm.Model() as model:
    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Logistic regression
    logits = pm.math.dot(X_norm, beta) + intercept
    p_malicious = pm.Deterministic("p_malicious", pm.math.sigmoid(logits))

    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=p_malicious, observed=y)

    # Sample from posterior
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95)

# -----------------------------
# 3. Predict new domain
# -----------------------------
# New domain features
new_domain = np.array([[120, 3, 0.85]])  # [ttl, num_clients, entropy]
new_domain_norm = (new_domain - X_mean) / X_std  # normalize using training stats

# Stack posterior samples
beta_samples = trace.posterior['beta'].stack(samples=("chain", "draw")).values  # shape: (features, samples)
intercept_samples = trace.posterior['intercept'].stack(samples=("chain", "draw")).values  # shape: (samples,)

# Compute sigmoid(beta⋅x + intercept) for each sample
logits = np.dot(beta_samples.T, new_domain_norm.T).squeeze() + intercept_samples
probs = 1 / (1 + np.exp(-logits))
malicious_prob = np.mean(probs)
ci_low, ci_high = np.percentile(probs, [2.5, 97.5])

print(f"Estimated probability of maliciousness: {malicious_prob:.3f}")
print(f"95% credible interval: ({ci_low:.3f}, {ci_high:.3f})")
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# -----------------------------
# 1. Sample input data (mock)
# -----------------------------
# You can replace this with your real feature data
data = pd.DataFrame({
    "ttl": [60, 300, 100, 1800, 50],
    "num_clients": [2, 10, 1, 20, 1],
    "entropy": [0.9, 0.5, 0.95, 0.4, 0.97],
    "ti_flag": [1, 0, None, 0, 1]  # External labels: 1 = malicious, 0 = benign
})

# Keep only rows with external evidence
observed_data = data.dropna(subset=["ti_flag"])
X = observed_data[["ttl", "num_clients", "entropy"]].values
y = observed_data["ti_flag"].values.astype(int)

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# -----------------------------
# 2. Build Bayesian Model
# -----------------------------
with pm.Model() as model:
    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Logistic regression
    logits = pm.math.dot(X_norm, beta) + intercept
    p_malicious = pm.Deterministic("p_malicious", pm.math.sigmoid(logits))

    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=p_malicious, observed=y)

    # Sample from posterior
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95)

# -----------------------------
# 3. Predict new domain
# -----------------------------
# New domain features
new_domain = np.array([[120, 3, 0.85]])  # [ttl, num_clients, entropy]
new_domain_norm = (new_domain - X_mean) / X_std  # normalize using training stats

# Stack posterior samples
beta_samples = trace.posterior['beta'].stack(samples=("chain", "draw")).values  # shape: (features, samples)
intercept_samples = trace.posterior['intercept'].stack(samples=("chain", "draw")).values  # shape: (samples,)

# Compute sigmoid(beta⋅x + intercept) for each sample
logits = np.dot(beta_samples.T, new_domain_norm.T).squeeze() + intercept_samples
probs = 1 / (1 + np.exp(-logits))
malicious_prob = np.mean(probs)
ci_low, ci_high = np.percentile(probs, [2.5, 97.5])

print(f"Estimated probability of maliciousness: {malicious_prob:.3f}")
print(f"95% credible interval: ({ci_low:.3f}, {ci_high:.3f})")

