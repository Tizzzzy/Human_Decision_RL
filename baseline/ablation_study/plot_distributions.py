"""
Plot distributions for human-preferred and model-preferred explanations separately.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# File paths
HUMAN_FILE = "human_preferred_explanations.json"
MODEL_FILE = "preference_pairs_aggregated.json"
OUTPUT_DIR = "distribution_plots"

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
# Figure 1: Human Preferences Distribution
# ==========================================
print("\nGenerating human preferences distributions...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Human-Preferred Explanations Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)

# Extract data
human_data = []
for key, val in human_prefs.items():
    human_data.append({
        'key': key,
        'label': val.get('label'),
        'chosen_source': val.get('chosen_explanation_source'),
        'accuracy': val.get('chosen_accuracy'),
        'decisions_count': val.get('chosen_decisions_count'),
        'decision_method': val.get('decision_method')
    })

df_human = pd.DataFrame(human_data)

# 1.1: Overall count
ax = fig.add_subplot(gs[0, 0])
label_counts = df_human['label'].value_counts()
colors = ['#3498db', '#e67e22']
ax.bar(label_counts.index, label_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Total Entries by Label', fontweight='bold')
for i, v in enumerate(label_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# 1.2: Accuracy distribution (overall)
ax = fig.add_subplot(gs[0, 1])
ax.hist(df_human['accuracy'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax.axvline(df_human['accuracy'].mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {df_human["accuracy"].mean():.3f}')
ax.axvline(df_human['accuracy'].median(), color='blue', linestyle='--', linewidth=2.5, label=f'Median: {df_human["accuracy"].median():.3f}')
ax.set_xlabel('Accuracy', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Accuracy Distribution (All)', fontweight='bold')
ax.legend()

# 1.3: Decision count distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(df_human['decisions_count'], bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Human Decisions', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Human Decisions per Entry', fontweight='bold')

# 1.4: Source distribution (overall)
ax = fig.add_subplot(gs[1, 0])
sources_human = df_human['chosen_source'].value_counts().sort_values(ascending=False)
ax.barh(range(len(sources_human)), sources_human.values, color='#1abc9c', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(sources_human)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_human.index], fontsize=9)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Chosen Sources (Overall)', fontweight='bold')
for i, v in enumerate(sources_human.values):
    ax.text(v, i, f' {v}', va='center', fontweight='bold')

# 1.5: Accuracy by label
ax = fig.add_subplot(gs[1, 1])
human_acc_by_label = df_human.groupby('label')['accuracy'].apply(list).to_dict()
bp = ax.boxplot([human_acc_by_label.get('Human', []), human_acc_by_label.get('AI', [])],
                 labels=['Human', 'AI'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Accuracy Distribution by Label', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1.6: Source by label (Human)
ax = fig.add_subplot(gs[1, 2])
human_texts = df_human[df_human['label'] == 'Human']['chosen_source'].value_counts()
ax.barh(range(len(human_texts)), human_texts.values, color='#3498db', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(human_texts)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in human_texts.index], fontsize=8)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Sources for Human Texts', fontweight='bold')

# 1.7: Source by label (AI)
ax = fig.add_subplot(gs[2, 0])
ai_texts = df_human[df_human['label'] == 'AI']['chosen_source'].value_counts()
ax.barh(range(len(ai_texts)), ai_texts.values, color='#e67e22', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(ai_texts)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in ai_texts.index], fontsize=8)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Sources for AI Texts', fontweight='bold')

# 1.8: Decision method distribution
ax = fig.add_subplot(gs[2, 1])
decision_methods = df_human['decision_method'].value_counts()
colors_methods = plt.cm.Set3(np.linspace(0, 1, len(decision_methods)))
ax.barh(range(len(decision_methods)), decision_methods.values, color=colors_methods, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(decision_methods)))
ax.set_yticklabels([m.replace('_', '\n') for m in decision_methods.index], fontsize=8)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Decision Methods Used', fontweight='bold')

# 1.9: Accuracy percentile
ax = fig.add_subplot(gs[2, 2])
accuracy_ranges = pd.cut(df_human['accuracy'], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0'])
accuracy_dist = accuracy_ranges.value_counts().sort_index()
ax.bar(range(len(accuracy_dist)), accuracy_dist.values, color='#f39c12', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(accuracy_dist)))
ax.set_xticklabels(accuracy_dist.index, rotation=45)
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Accuracy Range Distribution', fontweight='bold')

plt.savefig(f'{OUTPUT_DIR}/01_human_preferences_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_human_preferences_distribution.png")
plt.close()

# ==========================================
# Figure 2: Model Preferences Distribution
# ==========================================
print("Generating model preferences distributions...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Model-Preferred Explanations Distribution Analysis (Aggregated)', fontsize=18, fontweight='bold', y=0.995)

# Extract data
model_data = []
for key, val in model_prefs.items():
    model_data.append({
        'key': key,
        'label': val.get('label'),
        'chosen_source': val.get('chosen_source'),
        'winning_votes': val.get('winning_votes'),
        'total_votes': val.get('total_votes'),
        'model_votes': val.get('model_votes')
    })

df_model = pd.DataFrame(model_data)

# 2.1: Overall count
ax = fig.add_subplot(gs[0, 0])
label_counts = df_model['label'].value_counts()
colors = ['#3498db', '#e67e22']
ax.bar(label_counts.index, label_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Total Entries by Label', fontweight='bold')
for i, v in enumerate(label_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# 2.2: Winning votes distribution
ax = fig.add_subplot(gs[0, 1])
vote_dist = df_model['winning_votes'].value_counts().sort_index()
colors_votes = ['#e74c3c', '#f39c12', '#2ecc71']
ax.bar(vote_dist.index, vote_dist.values, color=colors_votes[:len(vote_dist)], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Winning Votes (out of 3)', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Model Voting Agreement Distribution', fontweight='bold')
ax.set_xticks(sorted(vote_dist.index))
for i, (vote, count) in enumerate(zip(vote_dist.index, vote_dist.values)):
    ax.text(vote, count, str(count), ha='center', va='bottom', fontweight='bold')

# 2.3: Unanimity percentage
ax = fig.add_subplot(gs[0, 2])
unanimous = (df_model['winning_votes'] == 3).sum()
not_unanimous = (df_model['winning_votes'] < 3).sum()
sizes = [unanimous, not_unanimous]
labels_pie = [f'Unanimous\n(3/3)\n{unanimous}', f'Not Unanimous\n(2/3)\n{not_unanimous}']
colors_pie = ['#2ecc71', '#f39c12']
ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops={'fontweight': 'bold'})
ax.set_title('Model Consensus Level', fontweight='bold')

# 2.4: Source distribution (overall)
ax = fig.add_subplot(gs[1, 0])
sources_model = df_model['chosen_source'].value_counts().sort_values(ascending=False)
ax.barh(range(len(sources_model)), sources_model.values, color='#16a085', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(sources_model)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_model.index], fontsize=9)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Chosen Sources (Overall)', fontweight='bold')
for i, v in enumerate(sources_model.values):
    ax.text(v, i, f' {v}', va='center', fontweight='bold')

# 2.5: Winning votes by label
ax = fig.add_subplot(gs[1, 1])
votes_by_label = df_model.groupby('label')['winning_votes'].value_counts().unstack(fill_value=0)
votes_by_label.plot(kind='bar', ax=ax, color=['#f39c12', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontweight='bold')
ax.set_xlabel('Label', fontweight='bold')
ax.set_title('Model Votes by Label', fontweight='bold')
ax.legend(title='Winning Votes', labels=['2/3', '3/3'])
ax.set_xticklabels(['AI', 'Human'], rotation=0)

# 2.6: Source by label (Human)
ax = fig.add_subplot(gs[1, 2])
human_texts_model = df_model[df_model['label'] == 'Human']['chosen_source'].value_counts()
ax.barh(range(len(human_texts_model)), human_texts_model.values, color='#3498db', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(human_texts_model)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in human_texts_model.index], fontsize=8)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Sources for Human Texts', fontweight='bold')

# 2.7: Source by label (AI)
ax = fig.add_subplot(gs[2, 0])
ai_texts_model = df_model[df_model['label'] == 'AI']['chosen_source'].value_counts()
ax.barh(range(len(ai_texts_model)), ai_texts_model.values, color='#e67e22', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(ai_texts_model)))
ax.set_yticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in ai_texts_model.index], fontsize=8)
ax.set_xlabel('Count', fontweight='bold')
ax.set_title('Sources for AI Texts', fontweight='bold')

# 2.8: Model agreement heatmap
ax = fig.add_subplot(gs[2, 1])
agreement_summary = df_model.groupby(['label', 'winning_votes']).size().unstack(fill_value=0)
im = ax.imshow(agreement_summary.values, cmap='YlGn', aspect='auto')
ax.set_xticks(range(len(agreement_summary.columns)))
ax.set_yticks(range(len(agreement_summary.index)))
ax.set_xticklabels(agreement_summary.columns)
ax.set_yticklabels(agreement_summary.index)
ax.set_xlabel('Winning Votes', fontweight='bold')
ax.set_ylabel('Label', fontweight='bold')
ax.set_title('Agreement Heatmap', fontweight='bold')
for i in range(len(agreement_summary.index)):
    for j in range(len(agreement_summary.columns)):
        text = ax.text(j, i, int(agreement_summary.values[i, j]),
                      ha="center", va="center", color="black", fontweight='bold')
plt.colorbar(im, ax=ax, label='Count')

# 2.9: Model voting agreement percentage
ax = fig.add_subplot(gs[2, 2])
vote_pct = (df_model['winning_votes'].value_counts() / len(df_model) * 100).sort_index()
ax.bar(vote_pct.index, vote_pct.values, color=['#f39c12', '#2ecc71'][:len(vote_pct)],
       alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Winning Votes (out of 3)', fontweight='bold')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Model Agreement Rate', fontweight='bold')
ax.set_xticks(sorted(vote_pct.index))
for vote, pct in zip(vote_pct.index, vote_pct.values):
    ax.text(vote, pct, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.savefig(f'{OUTPUT_DIR}/02_model_preferences_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_model_preferences_distribution.png")
plt.close()

# ==========================================
# Figure 3: Side-by-Side Source Comparison
# ==========================================
print("Generating source comparison visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Explanation Source Selection: Human vs Model Comparison', fontsize=16, fontweight='bold')

sources_all = sorted(set(df_human['chosen_source'].unique()) | set(df_model['chosen_source'].unique()))

# 3.1: Overall sources
ax = axes[0, 0]
human_source_counts = df_human['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
model_source_counts = df_model['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
x = np.arange(len(sources_all))
width = 0.35
ax.bar(x - width/2, human_source_counts.values, width, label='Human', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, model_source_counts.values, width, label='Model', alpha=0.8, edgecolor='black')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Source Selection: Overall', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_all], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3.2: Human texts only
ax = axes[0, 1]
human_human = df_human[df_human['label'] == 'Human']['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
model_human = df_model[df_model['label'] == 'Human']['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
x = np.arange(len(sources_all))
ax.bar(x - width/2, human_human.values, width, label='Human Judgment', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, model_human.values, width, label='Model Aggregation', alpha=0.8, edgecolor='black')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Source Selection: Human Texts Only', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_all], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3.3: AI texts only
ax = axes[1, 0]
human_ai = df_human[df_human['label'] == 'AI']['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
model_ai = df_model[df_model['label'] == 'AI']['chosen_source'].value_counts().reindex(sources_all, fill_value=0)
x = np.arange(len(sources_all))
ax.bar(x - width/2, human_ai.values, width, label='Human Judgment', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, model_ai.values, width, label='Model Aggregation', alpha=0.8, edgecolor='black')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Source Selection: AI Texts Only', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_all], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3.4: Source preference ratio
ax = axes[1, 1]
human_ratio = (human_source_counts / human_source_counts.sum() * 100).fillna(0)
model_ratio = (model_source_counts / model_source_counts.sum() * 100).fillna(0)
x = np.arange(len(sources_all))
ax.bar(x - width/2, human_ratio.values, width, label='Human', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, model_ratio.values, width, label='Model', alpha=0.8, edgecolor='black')
ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Source Selection: Percentage Distribution', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SocialMedia_', '').replace('_', '\n') for s in sources_all], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_source_selection_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_source_selection_comparison.png")
plt.close()

# ==========================================
# Summary Statistics
# ==========================================
summary = f"""
{'='*80}
DISTRIBUTION ANALYSIS SUMMARY
{'='*80}

HUMAN PREFERENCES ({len(df_human)} entries):
  Label distribution:
    - Human texts: {(df_human['label'] == 'Human').sum()}
    - AI texts: {(df_human['label'] == 'AI').sum()}

  Accuracy statistics:
    - Mean: {df_human['accuracy'].mean():.3f}
    - Median: {df_human['accuracy'].median():.3f}
    - Std Dev: {df_human['accuracy'].std():.3f}
    - Min: {df_human['accuracy'].min():.3f}
    - Max: {df_human['accuracy'].max():.3f}
    - Perfect accuracy (1.0): {(df_human['accuracy'] == 1.0).sum()} ({(df_human['accuracy'] == 1.0).sum()/len(df_human)*100:.1f}%)

  Decisions per entry:
    - Mean: {df_human['decisions_count'].mean():.2f}
    - Median: {df_human['decisions_count'].median():.0f}
    - Range: {df_human['decisions_count'].min():.0f} - {df_human['decisions_count'].max():.0f}

  Most preferred sources:
{chr(10).join([f'    - {src}: {cnt}' for src, cnt in df_human['chosen_source'].value_counts().head(3).items()])}

  Decision methods:
{chr(10).join([f'    - {method}: {cnt}' for method, cnt in df_human['decision_method'].value_counts().items()])}

MODEL PREFERENCES ({len(df_model)} entries):
  Label distribution:
    - Human texts: {(df_model['label'] == 'Human').sum()}
    - AI texts: {(df_model['label'] == 'AI').sum()}

  Model voting agreement:
    - Unanimous (3/3): {(df_model['winning_votes'] == 3).sum()} ({(df_model['winning_votes'] == 3).sum()/len(df_model)*100:.1f}%)
    - Majority (2/3): {(df_model['winning_votes'] == 2).sum()} ({(df_model['winning_votes'] == 2).sum()/len(df_model)*100:.1f}%)
    - Split (1/3): {(df_model['winning_votes'] == 1).sum()} ({(df_model['winning_votes'] == 1).sum()/len(df_model)*100:.1f}%)

  Most preferred sources:
{chr(10).join([f'    - {src}: {cnt}' for src, cnt in df_model['chosen_source'].value_counts().head(3).items()])}

KEY OBSERVATIONS:
1. Human preferences have lower coverage ({len(df_human)}) vs model preferences ({len(df_model)})
   → {len(df_model) - len(df_human)} entries have no human decision data (below 0.5 accuracy threshold)

2. Human judgment shows median perfect accuracy (1.0)
   → Humans are quite confident in their selections when they have enough data

3. Models show very high consensus
   → {(df_model['winning_votes'] == 3).sum()/len(df_model)*100:.1f}% unanimous agreement across 3 models

4. Source preferences differ:
   - Humans prefer: {df_human['chosen_source'].value_counts().index[0]}
   - Models prefer: {df_model['chosen_source'].value_counts().index[0]}

{'='*80}
"""

print(summary)

with open(f'{OUTPUT_DIR}/distribution_summary.txt', 'w') as f:
    f.write(summary)
print(f"✓ Saved: distribution_summary.txt")

print(f"\n✓ All distribution plots saved to: {OUTPUT_DIR}/")
