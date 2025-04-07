# Bayesian DNS Domain Anomaly Detection

A fully **Bayesian, unsupervised anomaly detection system** for DNS domains, using PyMC for inference, ArviZ for diagnostics, and well-explained feature modeling.

---

## 🔍 What This Project Does

This system models **maliciousness of DNS domains** based on features extracted every hour (aggregated from DNS logs).

We use a **Bayesian generative model** with latent variables to estimate the posterior probability that a domain is malicious — even without labeled data.

---

## 🧠 How the Bayesian Model Works

Each domain is modeled as coming from one of two unknown groups:

- 🟢 Benign
- 🔴 Malicious

Since we don’t have ground truth, we treat the class as a **latent variable**:

```
θ ~ Beta(1,1)                        # prior belief of global malicious rate
malicious[i] ~ Bernoulli(θ)         # latent variable per domain
```

Features depend on whether the domain is malicious or not:

```
X[i] ~ Normal(mu_0, sigma) if benign
X[i] ~ Normal(mu_1, sigma) if malicious
```

We model the means (`mu_0`, `mu_1`) and std (`sigma`) of the features as random variables too.

---

## 🔄 Learning Process

1. **Extract features** per domain per hour
2. **Standardize** them using mean/std
3. Use PyMC to build the **joint probabilistic model**
4. Sample from the **posterior using NUTS**
5. Infer:
   - `P(domain is malicious)`
   - `mu_0`, `mu_1` → how malicious/benign domains tend to behave
6. Use `arviz` to:
   - Diagnose convergence
   - Plot and interpret results

---

## 🔗 Diagram

```
           θ ~ Beta(1,1)
              ↓
     malicious[i] ~ Bernoulli(θ)
              ↓
        ┌──────────────┐
        │   Feature X  │
        └──────────────┘
              ↑
     ┌─────────────────────────┐
     │ if malicious[i] == 0 → N(mu_0, σ)
     │ if malicious[i] == 1 → N(mu_1, σ)
     └─────────────────────────┘
```

---

## 🧪 Feature List

The model expects 21 normalized features like:

- `num_requests`, `avg_ttl`, `ttl_range`, `ttl_entropy`
- `num_ips`, `ips_entropy`, `ip_sharing_count`
- FFT features: `dominant_frequency`, `spectral_entropy`
- Flags: `is_in_TI`, `is_in_tranco`

See the scripts for full details.

---

## 📁 Components

- `domain-bayes-unsup.py` — Training + inference + diagnostics + plots
- `infer_new_domain.py` — Predicts `P(malicious)` for new domains using trained posterior
- `batch_infer.py` — Predicts for many domains in a CSV
- `dns_bayesian_unsup_analysis.ipynb` — Jupyter version for exploration
- `requirements.txt` — All dependencies
- `maliciousness_report.csv` — Predictions
- `run_log.md` — Describes each run (timestamped)

---

## 🏃‍♂️ How to Use It

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python domain-bayes-unsup.py --output_dir results/YYYY-MM-DD
```

### 3. Inference on a new domain
```bash
python infer_new_domain.py --num_requests=... --min_ttl=... ...
```

### 4. Batch inference on a CSV
```bash
python batch_infer.py
```

---

## 📅 Run Log

This README was generated on: **2025-04-07 19:51:57**

---

## ✅ Advantages of This Bayesian Approach

- No labels needed — fully unsupervised
- Produces uncertainty + confidence intervals
- Highly interpretable
- Probabilistic scores (not black-box classifications)
- Feature weights learned from data

---

## 🛠 Future Ideas

- Online / streaming inference with dynamic priors
- Custom priors based on domain heuristics
- Hierarchical model across networks or tenants
- Time-evolving beliefs

---

## 🙌 Credits

Developed with ❤️ using PyMC, ArviZ, and NumPy.
