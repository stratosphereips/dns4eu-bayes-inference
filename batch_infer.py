
import numpy as np
import pandas as pd
import argparse
import pymc as pm
import arviz as az
import ast
import re
from scipy.stats import norm
from collections import Counter
from scipy.stats import entropy as shannon_entropy
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="CSV with multiple domains")
parser.add_argument("--trace_file", type=str, required=True, help="Trace file (.nc) from ADVI")
parser.add_argument("--mean_std_dir", type=str, required=True, help="Directory with X_mean.npy and X_std.npy")
parser.add_argument("--output_file", type=str, default="predictions.csv", help="Where to save output predictions")
args = parser.parse_args()

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
    return [float(i) for i in re.findall(r"\d+(?:\.\d+)?", str(x))]

def safe_entropy(seq):
    if not seq:
        return 0.0
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return shannon_entropy(probs, base=2)

def safe_range(x):
    x = list(x)
    return max(x) - min(x) if len(x) > 1 else 0.0

def iqr(x):
    x = list(x)
    if len(x) < 2:
        return 0.0
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

start_time = time.time()
original_df = pd.read_csv(args.input_file)
df = original_df.copy()

if "ttls" in df.columns:
    df["ttls"] = df["ttls"].apply(parse_list_safely)
    df["ttl_range"] = df["ttls"].apply(safe_range)
    df["ttl_entropy"] = df["ttls"].apply(safe_entropy)
    df["ttl_iqr"] = df["ttls"].apply(iqr)

if "answer_ips" in df.columns:
    df["answer_ips"] = df["answer_ips"].apply(parse_list_safely)
    df["ips_entropy"] = df["answer_ips"].apply(safe_entropy)
    df["ips_count"] = df["answer_ips"].apply(lambda x: len(set(x)))

for f in ["in_gmm_cluster", "in_isoforest_cluster", "in_dbscan_cluster"]:
    if f not in df.columns:
        df[f] = 0
if "is_in_TI" not in df.columns:
    df["is_in_TI"] = 0
if "is_in_tranco" not in df.columns:
    df["is_in_tranco"] = 0

X_mean = np.load(f"{args.mean_std_dir}/X_mean.npy")
X_std = np.load(f"{args.mean_std_dir}/X_std.npy")
X_std[X_std == 0] = 1

all_possible_features = [
    "num_requests", "min_ttl", "max_ttl", "avg_ttl", "stddev_ttl",
    "num_ips", "dominant_frequency", "total_power", "peak_magnitude",
    "mean_magnitude", "spectral_entropy", "ip_sharing_count",
    "ttl_unique_count", "domain_entropy", "is_in_TI", "is_in_tranco",
    "ttl_range", "ttl_entropy", "ttl_iqr", "ips_entropy", "ips_count",
    "in_gmm_cluster", "in_isoforest_cluster", "in_dbscan_cluster"
]
features = [f for f in all_possible_features if f in df.columns][:X_mean.shape[0]]
X = df[features].fillna(0).astype(float).values
X_norm = (X - X_mean) / X_std

trace = az.from_netcdf(args.trace_file)
theta_samples_raw = trace.posterior["theta"].stack(sample=("chain", "draw")).values.flatten()
mu_0_samples = trace.posterior["mu_0"].stack(sample=("chain", "draw")).values
mu_1_samples = trace.posterior["mu_1"].stack(sample=("chain", "draw")).values
sigma_samples = trace.posterior["sigma"].stack(sample=("chain", "draw")).values

S = mu_0_samples.shape[1]
theta_1 = theta_samples_raw[:S]
theta_0 = 1 - theta_1

X_norm = X_norm.astype(np.float32)
mu_0_samples = mu_0_samples[:, :S].astype(np.float32).T
mu_1_samples = mu_1_samples[:, :S].astype(np.float32).T
sigma_samples = sigma_samples[:, :S].astype(np.float32).T

def compute_log_probs(X, mu, sigma):
    part1 = -0.5 * np.sum(((X[:, None, :] - mu[None, :, :]) / sigma[None, :, :]) ** 2, axis=2)
    part2 = -np.sum(np.log(np.sqrt(2 * np.pi) * sigma), axis=1)
    return part1 + part2

log_probs_malicious = compute_log_probs(X_norm, mu_1_samples, sigma_samples)
log_probs_benign = compute_log_probs(X_norm, mu_0_samples, sigma_samples)
log_theta_ratio = np.log(theta_1) - np.log(theta_0)

log_ratios = log_probs_malicious + log_theta_ratio - log_probs_benign
prob_malicious = (log_ratios > 0).mean(axis=1)

original_df["prob_malicious"] = prob_malicious
original_df.to_csv(args.output_file, index=False)

elapsed = time.time() - start_time
print(f"✅ Saved predictions to {args.output_file}")
print(f"⏱️ Total processing time: {elapsed:.2f} seconds")
