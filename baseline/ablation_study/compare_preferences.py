"""
Compare and analyze human-preferred vs model-preferred explanations.

This script:
1. Loads both preference files
2. Compares which explanations humans and models chose
3. Calculates agreement rates and statistics
4. Generates visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np

# File paths
HUMAN_FILE = "human_preferred_explanations.json"
MODEL_FILE = "preference_pairs_aggregated.json"
OUTPUT_DIR = "comparison_analysis"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load Data
# ==========================================
print("Loading preference files...")
with open(HUMAN_FILE, 'r') as f:
    human_prefs = json.load(f)
print(f"Human preferences: {len(human_prefs)} entries")

with open(MODEL_FILE, 'r') as f:
    model_prefs = json.load(f)
print(f"Model preferences: {len(model_prefs)} entries")

# ==========================================
# Match and Compare
# ==========================================
print("\nMatching and comparing preferences...")

matches = 0
mismatches = 0
human_only = 0
model_only = 0

agreement_data = []
source_comparison = defaultdict(lambda: {'human': 0, 'model': 0, 'both': 0})

for key in human_prefs.keys():
    if key not in model_prefs:
        human_only += 1
        continue

    human = human_prefs[key]
    model = model_prefs[key]

    human_source = human.get('chosen_explanation_source')
    model_source = model.get('chosen_source')

    # Check if they agree
    agree = human_source == model_source
    if agree:
        matches += 1
    else:
        mismatches += 1

    # Track source preferences
    source_comparison[human_source]['human'] += 1
    source_comparison[model_source]['model'] += 1
    if human_source == model_source:
        source_comparison[human_source]['both'] += 1

    # Store comparison data
    agreement_data.append({
        'text_id': key,
        'label': human.get('label'),
        'human_source': human_source,
        'model_source': model_source,
        'human_accuracy': human.get('chosen_accuracy'),  # Updated field name
        'human_decisions': human.get('chosen_decisions_count'),  # Updated field name
        'model_votes': model.get('winning_votes'),
        'model_vote_agreement': model.get('total_votes'),
        'agree': agree
    })

for key in model_prefs.keys():
    if key not in human_prefs:
        model_only += 1

df_comparison = pd.DataFrame(agreement_data)

# ==========================================
# Analysis Statistics
# ==========================================
print("\n" + "="*80)
print("COMPARISON ANALYSIS")
print("="*80)

print(f"\nData Coverage:")
print(f"  Entries in both files: {matches + mismatches}")
print(f"  Human only: {human_only}")
print(f"  Model only: {model_only}")
print(f"  Total human entries: {len(human_prefs)}")
print(f"  Total model entries: {len(model_prefs)}")

total_comparable = matches + mismatches
if total_comparable > 0:
    agreement_rate = (matches / total_comparable) * 100
    print(f"\nAgreement Statistics:")
    print(f"  Matches: {matches} ({agreement_rate:.1f}%)")
    print(f"  Mismatches: {mismatches} ({100-agreement_rate:.1f}%)")

# By label
print(f"\nAgreement by Label:")
for label in ['Human', 'AI']:
    label_data = df_comparison[df_comparison['label'] == label]
    if len(label_data) > 0:
        label_agreement = label_data['agree'].sum() / len(label_data) * 100
        print(f"  {label}: {label_agreement:.1f}% ({label_data['agree'].sum()}/{len(label_data)})")

# ==========================================
# Source Comparison
# ==========================================
print(f"\nExplanation Source Preferences:")
print(f"\n{'Source':<40} {'Human':<10} {'Model':<10} {'Agreement':<10}")
print("-" * 70)
for source in sorted(source_comparison.keys()):
    h = source_comparison[source]['human']
    m = source_comparison[source]['model']
    b = source_comparison[source]['both']
    print(f"{source:<40} {h:<10} {m:<10} {b:<10}")

# ==========================================
# Detailed Mismatch Analysis
# ==========================================
print(f"\n" + "="*80)
print("MISMATCH ANALYSIS (Where human and model disagree)")
print("="*80)

mismatches_df = df_comparison[~df_comparison['agree']]
if len(mismatches_df) > 0:
    print(f"\nTotal mismatches: {len(mismatches_df)}")

    # Most common disagreements
    disagreement_pairs = mismatches_df.groupby(['human_source', 'model_source']).size().reset_index(name='count')
    disagreement_pairs = disagreement_pairs.sort_values('count', ascending=False)

    print(f"\nMost common disagreements (Human → Model):")
    for idx, row in disagreement_pairs.head(10).iterrows():
        print(f"  {row['human_source']} → {row['model_source']}: {row['count']} cases")

    # Accuracy analysis of mismatches
    avg_human_acc = mismatches_df['human_accuracy'].mean()
    avg_model_agreement = mismatches_df['model_votes'].mean()
    print(f"\nMismatch quality:")
    print(f"  Avg human accuracy in mismatches: {avg_human_acc:.3f}")
    print(f"  Avg model voting agreement: {avg_model_agreement:.2f}/3")

# ==========================================
# Generate Visualizations
# ==========================================
print(f"\nGenerating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Figure 1: Overall Agreement
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Human vs Model Preference Agreement Analysis', fontsize=16, fontweight='bold')

# 1.1: Overall agreement pie chart
ax = axes[0, 0]
agreement_counts = [matches, mismatches]
colors = ['#2ecc71', '#e74c3c']
ax.pie(agreement_counts, labels=['Agreement', 'Disagreement'], autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 11})
ax.set_title('Overall Agreement Rate', fontweight='bold')

# 1.2: Agreement by label
ax = axes[0, 1]
labels = ['Human', 'AI']
agreements = []
for label in labels:
    label_data = df_comparison[df_comparison['label'] == label]
    if len(label_data) > 0:
        pct = label_data['agree'].sum() / len(label_data) * 100
        agreements.append(pct)
    else:
        agreements.append(0)

bars = ax.bar(labels, agreements, color=['#3498db', '#e67e22'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
ax.set_title('Agreement by Text Label', fontweight='bold')
ax.set_ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# 1.3: Source preference comparison
ax = axes[1, 0]
sources = sorted(source_comparison.keys())
human_counts = [source_comparison[s]['human'] for s in sources]
model_counts = [source_comparison[s]['model'] for s in sources]

x = np.arange(len(sources))
width = 0.35
ax.bar(x - width/2, human_counts, width, label='Human', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, model_counts, width, label='Model', alpha=0.8, edgecolor='black')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Source Preference Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources],
                     fontsize=9, rotation=0)
ax.legend()

# 1.4: Mismatch rate by label
ax = axes[1, 1]
if len(mismatches_df) > 0:
    mismatch_by_label = mismatches_df['label'].value_counts()
    labels_list = ['Human', 'AI']
    mismatch_counts = [mismatch_by_label.get(l, 0) for l in labels_list]
    bars = ax.bar(labels_list, mismatch_counts, color=['#e74c3c', '#c0392b'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Mismatches', fontweight='bold')
    ax.set_title('Mismatch Count by Label', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_agreement_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_agreement_analysis.png")
plt.close()

# Figure 2: Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Preference Distribution Analysis', fontsize=16, fontweight='bold')

# 2.1: Human accuracy distribution
ax = axes[0, 0]
ax.hist(df_comparison['human_accuracy'].dropna(), bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax.set_xlabel('Accuracy', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Human Decision Accuracy Distribution', fontweight='bold')
ax.axvline(df_comparison['human_accuracy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_comparison["human_accuracy"].mean():.3f}')
ax.legend()

# 2.2: Human decisions count distribution
ax = axes[0, 1]
human_decisions_valid = df_comparison['human_decisions'].dropna()
if len(human_decisions_valid) > 0:
    ax.hist(human_decisions_valid, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    q95 = human_decisions_valid.quantile(0.95)
    if not np.isnan(q95) and q95 > 0:
        ax.set_xlim([0, q95])
ax.set_xlabel('Number of Human Decisions', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Human Decisions per Entry Distribution', fontweight='bold')

# 2.3: Model voting agreement
ax = axes[1, 0]
vote_counts = df_comparison['model_votes'].value_counts().sort_index()
ax.bar(vote_counts.index, vote_counts.values, color=['#e74c3c', '#f39c12', '#2ecc71'],
       alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Model Voting Agreement (out of 3)', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Model Consensus Distribution', fontweight='bold')
ax.set_xticks([1, 2, 3])
for i, v in vote_counts.items():
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# 2.4: Agreement by human accuracy bins
ax = axes[1, 1]
df_comparison['accuracy_bin'] = pd.cut(df_comparison['human_accuracy'],
                                       bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                       labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0'])
agreement_by_accuracy = df_comparison.groupby('accuracy_bin', observed=True)['agree'].agg(['sum', 'count'])
agreement_by_accuracy['rate'] = (agreement_by_accuracy['sum'] / agreement_by_accuracy['count'] * 100)

bars = ax.bar(range(len(agreement_by_accuracy)), agreement_by_accuracy['rate'],
              color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Human Accuracy Range', fontweight='bold')
ax.set_ylabel('Agreement Rate with Model (%)', fontweight='bold')
ax.set_title('Model-Human Agreement vs Human Accuracy', fontweight='bold')
ax.set_xticks(range(len(agreement_by_accuracy)))
ax.set_xticklabels(agreement_by_accuracy.index, rotation=45)
ax.set_ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_distribution_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_distribution_analysis.png")
plt.close()

# Figure 3: Source-by-Source Comparison
fig, ax = plt.subplots(figsize=(12, 8))

sources = sorted(source_comparison.keys())
source_labels = [s.replace('SocialMedia_', '').replace('_', ' ') for s in sources]

human_data = [source_comparison[s]['human'] for s in sources]
model_data = [source_comparison[s]['model'] for s in sources]
both_data = [source_comparison[s]['both'] for s in sources]

x = np.arange(len(sources))
width = 0.25

bars1 = ax.bar(x - width, human_data, width, label='Chosen by Human', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, model_data, width, label='Chosen by Model', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, both_data, width, label='Agreement', alpha=0.8, edgecolor='black')

ax.set_ylabel('Count', fontweight='bold', fontsize=12)
ax.set_title('Explanation Source Preferences - Detailed Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(source_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_source_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_source_comparison.png")
plt.close()

# ==========================================
# Summary Report
# ==========================================
summary_report = f"""
{'='*80}
COMPARISON SUMMARY REPORT
{'='*80}

DATASET COVERAGE:
- Human preferences loaded: {len(human_prefs)} entries
- Model preferences loaded: {len(model_prefs)} entries
- Overlapping entries: {matches + mismatches}
- Agreement: {matches} ({(matches/(matches+mismatches)*100):.1f}%)
- Disagreement: {mismatches} ({(mismatches/(matches+mismatches)*100):.1f}%)

KEY FINDINGS:
1. Human and model agree on {(matches/(matches+mismatches)*100):.1f}% of entries
2. Most common disagreement:
   {disagreement_pairs.iloc[0]['human_source']} (Human) vs {disagreement_pairs.iloc[0]['model_source']} (Model)
   ({disagreement_pairs.iloc[0]['count']} cases)

HUMAN PREFERENCES:
- Average accuracy: {df_comparison['human_accuracy'].mean():.3f}
- Median accuracy: {df_comparison['human_accuracy'].median():.3f}
- Entries with perfect accuracy (1.0): {(df_comparison['human_accuracy'] == 1.0).sum()}
- Entries below threshold (< 0.5): 0 (filtered out)

MODEL PREFERENCES:
- Unanimous agreement (3/3): {(df_comparison['model_votes'] == 3).sum()}
- Majority agreement (2/3): {(df_comparison['model_votes'] == 2).sum()}
- Split decision (1/3): {(df_comparison['model_votes'] == 1).sum()}

ACCURACY VS AGREEMENT:
- Entries where human accuracy >= 0.8 and model-human agree: {((df_comparison['human_accuracy'] >= 0.8) & df_comparison['agree']).sum()}
- Entries where human accuracy < 0.8 and model-human agree: {((df_comparison['human_accuracy'] < 0.8) & df_comparison['agree']).sum()}

EXPLANATION SOURCES:
- Most preferred by humans: {max(source_comparison.keys(), key=lambda k: source_comparison[k]['human'])}
- Most preferred by models: {max(source_comparison.keys(), key=lambda k: source_comparison[k]['model'])}
- Highest agreement: {max(source_comparison.keys(), key=lambda k: source_comparison[k]['both'])}

{'='*80}
"""

print(summary_report)

# Save report
with open(f'{OUTPUT_DIR}/summary_report.txt', 'w') as f:
    f.write(summary_report)
print(f"\n✓ Saved: summary_report.txt")

# Save detailed comparison CSV
df_comparison.to_csv(f'{OUTPUT_DIR}/detailed_comparison.csv', index=False)
print(f"✓ Saved: detailed_comparison.csv")

print(f"\n✓ All analysis files saved to: {OUTPUT_DIR}/")
