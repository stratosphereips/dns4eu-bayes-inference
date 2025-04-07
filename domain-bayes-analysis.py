import pymc as pm
import numpy as np
import pandas as pd

# Sample input data
# X = features extracted from DNS traffic
# y_obs = external evidence (e.g., threat intel: 1 = flagged malicious, 0 = whitelist, None = unknown)
data = pd.DataFrame({
    "ttl": [60, 300, 100, 1800, 50],
    "num_clients": [2, 10, 1, 20, 1],
    "entropy": [0.9, 0.5, 0.95, 0.4, 0.97],
    "ti_flag": [1, 0, None, 0, 1]
})

# Only keep rows where external evidence exists (used to update belief)
observed_data = data.dropna(subset=["ti_flag"])
X = observed_data[["ttl", "num_clients", "entropy"]].values
y = observed_data["ti_flag"].values

# Normalize features (good practice)
X = (X - X.mean(axis=0)) / X.std(axis=0)

with pm.Model() as model:
    # Priors on regression coefficients
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Logistic regression
    logits = pm.math.dot(X, beta) + intercept
    p_malicious = pm.Deterministic("p_malicious", pm.math.sigmoid(logits))

    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=p_malicious, observed=y)

    # Inference
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95)

# Now we can use posterior samples to compute probabilities for new domains
from pymc.sampling import sample_posterior_predictive

# Example of new, unlabeled domain features
new_domain = np.array([[120, 3, 0.85]])
new_domain = (new_domain - X.mean(axis=0)) / X.std(axis=0)

with model:
    posterior_preds = pm.sample_posterior_predictive(trace, var_names=["beta", "intercept"])

# Compute predicted probability manually using posterior samples
logits = np.dot(posterior_preds["beta"], new_domain.T).squeeze() + posterior_preds["intercept"]
probs = 1 / (1 + np.exp(-logits))
malicious_prob = np.mean(probs)

print(f"Estimated probability of maliciousness: {malicious_prob:.3f}")
