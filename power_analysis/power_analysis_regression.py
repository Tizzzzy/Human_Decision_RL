import numpy as np
import pandas as pd
from scipy.special import expit # Sigmoid function to convert log-odds to probabilities
from pymer4.models import lmer  # Updated to 'lmer' to avoid deprecation warnings
import matplotlib.pyplot as plt

# 1. Base Parameters
# Convert empirical baseline (51.28%) to Log-Odds
# logit(p) = ln(p / (1-p)) -> ln(0.5128 / 0.4872) ≈ 0.051
baseline_log_odds = 0.051 

# Convert Qwen Simulator Target (71.03%) to Log-Odds
# ln(0.7103 / 0.2897) ≈ 0.896
# The "treatment effect" is the difference: 0.896 - 0.051 = 0.845
treatment_effect = 0.845 

# Estimated variances for the random effects (adjust based on your dataset)
var_human = 0.3  # Variance in how good different humans are at spotting AI
var_text = 0.5   # Variance in how difficult different texts are

# 2. Dataset Structure
# Fixed Baseline
n_base_decisions = 20520
n_base_texts = 3715
n_base_humans = int(n_base_decisions / 8.67) # ~2366 humans

# Deployment Targets
n_deploy_texts = 800
decisions_axis = range(800, 4800, 800) # Testing 1 to 5 decisions per text

# Note: GLMMs are computationally heavy. On a cluster like qgpu2003, 
# 100 trials is recommended over 1000 for standard power analyses to save time.
m_trials = 100 
power_results = []

print("Running Cross-Classified GLMM Monte Carlo Simulation...")

for n_new_decisions in decisions_axis:
    significant_count = 0
    n_new_humans = int(n_new_decisions / 8.67)
    
    for _ in range(m_trials):
        # --- A. Generate Random Intercepts (The Mixed Effects) ---
        # Baseline pool
        base_human_intercepts = np.random.normal(0, np.sqrt(var_human), n_base_humans)
        base_text_intercepts = np.random.normal(0, np.sqrt(var_text), n_base_texts)
        # Deployment pool
        new_human_intercepts = np.random.normal(0, np.sqrt(var_human), n_new_humans)
        new_text_intercepts = np.random.normal(0, np.sqrt(var_text), n_deploy_texts)
        
        # --- B. Assign IDs to Decisions ---
        base_h_ids = np.random.choice(range(n_base_humans), n_base_decisions)
        base_t_ids = np.random.choice(range(n_base_texts), n_base_decisions)
        
        new_h_ids = np.random.choice(range(n_new_humans), n_new_decisions)
        new_t_ids = np.random.choice(range(n_deploy_texts), n_new_decisions)
        
        # --- C. Calculate Log-Odds and Probabilities ---
        # Baseline: Intercept + Human Noise + Text Noise
        base_logits = baseline_log_odds + base_human_intercepts[base_h_ids] + base_text_intercepts[base_t_ids]
        base_probs = expit(base_logits)
        base_outcomes = np.random.binomial(1, base_probs)
        
        # Deployment: Intercept + Treatment Effect + Human Noise + Text Noise
        new_logits = baseline_log_odds + treatment_effect + new_human_intercepts[new_h_ids] + new_text_intercepts[new_t_ids]
        new_probs = expit(new_logits)
        new_outcomes = np.random.binomial(1, new_probs)
        
        # --- D. Build DataFrame ---
        df_base = pd.DataFrame({
            'outcome': base_outcomes,
            'treatment': 0,
            'human_id': [f"base_h_{i}" for i in base_h_ids],
            'text_id': [f"base_t_{i}" for i in base_t_ids]
        })
        
        df_new = pd.DataFrame({
            'outcome': new_outcomes,
            'treatment': 1,
            'human_id': [f"new_h_{i}" for i in new_h_ids],
            'text_id': [f"new_t_{i}" for i in new_t_ids]
        })
        
        df = pd.concat([df_base, df_new], ignore_index=True)
        
        # --- E. Fit the Mixed-Effects Model ---
        # We use a binomial family for logistic regression
        try:
            model = lmer("outcome ~ treatment + (1|human_id) + (1|text_id)", data=df, family='binomial')
            model.fit(summary=False)
            
            # Extract p-value for the treatment condition
            p_val = model.coefs.loc['treatment', 'P-val']
            
            if p_val < 0.05:
                significant_count += 1
        except Exception as e:
            # GLMMs occasionally fail to converge on highly simulated data; ignore and continue
            continue 
            
    power = significant_count / m_trials
    power_results.append(power)
    print(f"Decisions: {n_new_decisions} | Power: {power:.2%}")

# --- F. Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(decisions_axis, power_results, marker='o', linewidth=2.5, color='darkorange', label='Simulator Target (+20%)')
plt.axhline(y=0.80, color='red', linestyle='--', label='80% Power Threshold')

plt.title('Mixed-Effects GLMM Power Analysis (Cross-Classified)', fontsize=14)
plt.xlabel('Total NEW Decisions in Deployment', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.ylim(0, 1.05)
plt.xticks(decisions_axis, [f"{d}\n({int(d/800)}/text)" for d in decisions_axis])
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('power_analysis_regression.png', dpi=300)
# plt.show()