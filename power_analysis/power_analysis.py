import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 1. Exact Asymmetric Baseline Data
n_base_human = 10374
n_base_ai = 10146
base_acc_human = 0.6117
base_acc_ai = 0.4116

# 2. X-Axis Range (50 to 4000 decisions, stepping by 50)
decisions_axis = range(50, 4050, 50)

# 3. Your Target Accuracy Scenarios (Human Acc, AI Acc)
target_scenarios = {
    "AI +5% (Human 61.17%, AI 46.16%)": (0.6117, 0.4616),
    "AI +10% (Human 61.17%, AI 51.16%)": (0.6117, 0.5116),
    "AI +15% (Human 61.17%, AI 56.16%)": (0.6117, 0.5616),
    "AI +20% (Human 61.17%, AI 61.16%)": (0.6117, 0.6116),
    "Both Improve (Human 66.17%, AI 66.16%)": (0.6617, 0.6616)
}

m_trials = 1000
plt.figure(figsize=(12, 7))

print("Simulating power curves for independent accuracy shifts...")

for label, (tgt_human_acc, tgt_ai_acc) in target_scenarios.items():
    power_results = []
    for n_new in decisions_axis:
        # Final deployment assumes a balanced 50/50 text distribution
        n_new_human = n_new // 2
        n_new_ai = n_new - n_new_human
        
        significant_count = 0
        for _ in range(m_trials):
            # A. Simulate massive baseline pool
            base_h = np.random.binomial(1, base_acc_human, size=n_base_human)
            base_a = np.random.binomial(1, base_acc_ai, size=n_base_ai)
            baseline_group = np.concatenate([base_h, base_a])
            
            # B. Simulate new deployment data
            new_h = np.random.binomial(1, tgt_human_acc, size=n_new_human)
            new_a = np.random.binomial(1, tgt_ai_acc, size=n_new_ai)
            new_group = np.concatenate([new_h, new_a])
            
            # C. Run t-test
            t_stat, p_val = ttest_ind(baseline_group, new_group)
            
            if p_val < 0.05:
                significant_count += 1
                
        power_results.append(significant_count / m_trials)
        
    plt.plot(decisions_axis, power_results, linewidth=2.5, label=label)

# 4. Add visual thresholds
plt.axhline(y=0.80, color='red', linestyle='--', label='80% Power Threshold')
plt.axvline(x=800, color='gray', linestyle=':', label='1 Decision Per Text (N=800)')

# 5. Formatting
plt.title('Power Curves: Independent Human vs. AI Accuracy Shifts', fontsize=14)
plt.xlabel('Total NEW Decisions', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('power_analysis_independent_accuracy_shifts.png', dpi=300)