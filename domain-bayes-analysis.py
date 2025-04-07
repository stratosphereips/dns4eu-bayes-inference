import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import entropy as shannon_entropy
from collections import Counter

# ---------------------------------------------
# Function to compute entropy of a domain string
# ---------------------------------------------
def compute_entropy(domain):
    counts = Counter(domain)
    probs = np.array(list(counts.values())) / len(domain)
    return shannon_entropy(probs, base=2)

# ---------------------------------------------
# Load CSV data
# ---------------------------------------------
filename = "dns_data.csv"  # <--- Replace with your CSV path
data = pd.read_csv(filename)

# Compute entropy for each domain
data["entropy"] = data["domain"].apply(compute_entropy)

# Keep only rows with ti_flag to train the model
train_data = data.dropna(subset=["ti_flag"])
X = train_data[["TTL", "num_clients", "entropy"]].values
y = train_data["ti_flag"].astype(int).values

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# ---------------------------------------------
# Build and train the Bayesian logistic model
# ---------------------------------------------
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    logits = pm.math.dot(X_norm, beta) + intercept
    p_malicious = pm.Deterministic("p_malicious", pm.math.sigmoid(logits))
    
    y_obs = pm.Bernoulli("y_obs", p=p_malicious, observed=y)
    
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95, progressbar=True)

# Stack posterior samples
beta_samples = trace.posterior["beta"].stack(samples=("chain", "draw")).values
intercept_samples = trace.posterior["intercept"].stack(samples=("chain", "draw")).values

# ---------------------------------------------
# Predict for each domain in the dataset
# ---------------------------------------------
print("\nPredictions:")
print("Domain\t\t\tP(Malicious)\t95% CI")

for _, row in data.iterrows():
    features = np.array([[row["TTL"], row["num_clients"], row["entropy"]]])
    features_norm = (features - X_mean) / X_std
    
    logits = np.dot(beta_samples.T, features_norm.T).squeeze() + intercept_samples
    probs = 1 / (1 + np.exp(-logits))
    
    pred_mean = np.mean(probs)
    ci_low, ci_high = np.percentile(probs, [2.5, 97.5])
    
    print(f"{row['domain']:<24}{pred_mean:.3f}\t\t({ci_low:.3f}, {ci_high:.3f})")

