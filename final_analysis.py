#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import glob

plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

RES_DIR = '/workspace/task1/res'
RESULTS_DIR = '/workspace/task1/results'
INFERENCE_DIR = f'{RESULTS_DIR}/inference'
VIS_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIS_DIR, exist_ok=True)

print('='*80)
print('COMPREHENSIVE ANALYSIS - TASK 1')
print('='*80)

# Load inference results
print('\n1. Loading inference results...')
model_data = []
for pkl_file in glob.glob(f'{INFERENCE_DIR}/*.pkl'):
    with open(pkl_file, 'rb') as f:
      data = pickle.load(f)
    fn = os.path.basename(pkl_file)
    att = data.get('attention_type', 'Unknown')
    cp = data.get('checkpoint_path', '')
    mt = 'roberta' if 'roberta' in (cp + fn).lower() else 'bert'
    model_data.append({'model_type': mt, 'attention_type': att, 'data': data})
    print(f'  ✓ {mt}_{att}')

print(f'\nTotal: {len(model_data)} models')

# Parameter-Memory Analysis
print('\n' + '='*80)
print('2. PARAMETER-MEMORY RELATIONSHIP')
print('='*80)

params = []
memory = []
labels = []
for m in model_data:
    d = m['data']
    np_val = d.get('model_info', {}).get('num_parameters', 0)
    mem = d.get('performance_metrics', {}).get('peak_memory_gb', 0)
    if np_val > 0 and mem > 0:
        params.append(np_val / 1e6)
        memory.append(mem)
        labels.append(f"{m['model_type']}_{m['attention_type']}")

params = np.array(params)
memory = np.array(memory)
slope, intercept, r_val, p_val, std_err = stats.linregress(params, memory)

print(f'\nLinear Regression:')
print(f'  Memory (GB) = {slope:.4f} × Params (M) + {intercept:.2f}')
print(f'  R² = {r_val**2:.4f}, p-value = {p_val:.6f}')

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
colors = ['blue' if 'bert' in l else 'red' for l in labels]
markers = ['o' if 'MHA' in l else ('s' if 'GQA' in l else '^') for l in labels]
for x, y, l, c, m in zip(params, memory, labels, colors, markers):
    ax.scatter(x, y, c=c, marker=m, s=150, label=l, alpha=0.7, edgecolors='black', linewidths=2)
x_line = np.linspace(params.min(), params.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2.5, label=f'Linear Fit (R²={r_val**2:.3f})')
ax.text(0.05, 0.95, f'Memory = {slope:.4f} × Params + {intercept:.2f}', transform=ax.transAxes, fontsize=13, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.set_xlabel('Model Parameters (Millions)', fontsize=13, fontweight='bold')
ax.set_ylabel('Peak GPU Memory (GB)', fontsize=13, fontweight='bold')
ax.set_title('Linear Relationship: Parameters vs Memory', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f'{VIS_DIR}/parameter_memory_relationship.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: parameter_memory_relationship.png')
plt.close()

pd.DataFrame({'Model': labels, 'Parameters_M': params, 'Peak_Memory_GB': memory}).to_csv(f'{VIS_DIR}/parameter_memory_data.csv', index=False)
print(f'✓ Saved: parameter_memory_data.csv')

# Attention-Convergence Analysis
print('\n' + '='*80)
print('3. ATTENTION MECHANISM CONVERGENCE')
print('='*80)

conv_data = {}
for model_dir in os.listdir(RES_DIR):
    parts = model_dir.split('_')
    if len(parts) < 3:
        continue
    mt = parts[0]
    att = parts[2]
    ts_file = f'{RES_DIR}/{model_dir}/checkpoint/trainer_state.json'
    if os.path.exists(ts_file):
        with open(ts_file, 'r') as f:
            ts = json.load(f)
        log_hist = ts.get('log_history', [])
        steps = []
        losses = []
        for entry in log_hist:
            if 'loss' in entry:
                steps.append(entry.get('step', entry.get('epoch', 0)))
                losses.append(entry["loss"])
        if losses:
            conv_data[f'{mt}_{att}'] = {'steps': steps, 'losses': losses, 'model_type': mt, 'attention_type': att}
            print(f'  ✓ {mt}_{att}')

# Calculate metrics
metrics = []
for key, d in conv_data.items():
    losses = np.array(d['losses'])
    if len(losses) < 2:
        continue
    init_loss = losses[0]
    final_loss = losses[-1]
    loss_red = init_loss - final_loss
    conv_rate = loss_red / len(losses)
    target = init_loss - 0.9 * loss_red
    steps_90 = np.argmax(losses <= target) if np.any(losses <= target) else len(losses)
    metrics.append({
        'Model': key,
        'Model_Type': d['model_type'],
        'Attention_Type': d['attention_type'],
      'Initial_Loss': init_loss,
        'Final_Loss': final_loss,
        'Loss_Reduction': loss_red,
        'Convergence_Rate': conv_rate,
    'Steps_to_90%': steps_90,
        'Total_Steps': len(losses)
    })

df_metrics = pd.DataFrame(metrics)
print('\nConvergence Metrics:')
print(df_metrics.to_string(index=False))

# Plot curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, mt in enumerate(['bert', 'roberta']):
    ax = axes[idx]
    for key, d in conv_data.items():
        if d['model_type'] == mt:
          ax.plot(d['steps'], d['losses'], marker='o', label=d['attention_type'], linewidth=2.5, markersize=5, alpha=0.8)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{mt.upper()} - Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f'{VIS_DIR}/attention_convergence_curves.png', dpi=300, bbox_inches='tight')
print(f'\n✓ Saved: attention_convergence_curves.png')
plt.close()

# Plot rate comparison
fig, ax = plt.subplots(figsize=(10, 6))
att_groups = df_metrics.groupby('Attention_Type')['Convergence_Rate'].mean()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(att_groups.index, att_groups.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xlabel('Attention Mechanism', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Convergence Rate', fontsize=13, fontweight='bold')
ax.set_title('Attention Mechanism Impact on Convergence Speed', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(f'{VIS_DIR}/attention_convergence_rate.png', dpi=300, bbox_inches='tight')
print(f'✓ Saved: attention_convergence_rate.png')
plt.close()

df_metrics.to_csv(f'{VIS_DIR}/convergence_metrics.csv', index=False)
print(f'✓ Saved: convergence_metrics.csv')

# Generate report
print('\n' + '='*80)
print('4. GENERATING REPORT')
print('='*80)

report = []
report.append('='*80)
report.append('COMPREHENSIVE EVALUATION REPORT')
report.append('Task 1: Model Evaluation with Multiple Attention Mechanisms')
report.append('='*80)
report.append('')
report.append('1. MODEL PERFORMANCE SUMMARY')
report.append('-'*80)
perf_data = []
for m in model_data:
    p = m['data'].get('performance_metrics', {})
    perf_data.append({
        'Model': f"{m['model_type']}_{m['attention_type']}",
        'Accuracy': f"{p.get('accuracy', 0):.4f}",
      'F1': f"{p.get('f1_score', 0):.4f}",
        'Precision': f"{p.get('precision', 0):.4f}",
        'Recall': f"{p.get('recall', 0):.4f}",
        'AUC': f"{p.get('auc', 0):.4f}",
        'MCC': f"{p.get('mcc', 0):.4f}",
        'Memory_GB': f"{p.get('peak_memory_gb', 0):.2f}"
    })
report.append(pd.DataFrame(perf_data).to_string(index=False))
report.append('')
report.append('2. PARAMETER-MEMORY RELATIONSHIP')
report.append('-'*80)
report.append(f'Linear Regression: Memory (GB) = {slope:.4f} × Params (M) + {intercept:.2f}')
report.append(f'R² = {r_val**2:.4f}, p-value = {p_val:.6f}')
if r_val**2 > 0.9:
    report.append('✓ Strong linear relationship confirmed')
elif r_val**2 > 0.7:
    report.append('✓ Moderate linear relationship')
else:
    report.append('⚠ Weak linear relationship')
report.append('')
report.append('3. ATTENTION MECHANISM CONVERGENCE')
report.append('-'*80)
report.append(df_metrics.to_string(index=False))
report.append('')
best = df_metrics.loc[df_metrics['Convergence_Rate'].idxmax()]
report.append(f"✓ Fastest Convergence: {best['Attention_Type']} (Rate: {best['Convergence_Rate']:.4f})")
report.append('Average Convergence Rates:')
for att, rate in att_groups.items():
    report.append(f'  {att}: {rate:.4f}')
report.append('')
report.append('='*80)
report.append('END OF REPORT')
report.append('='*80)

report_text = '\n'.join(report)
with open(f'{RESULTS_DIR}/COMPREHENSIVE_EVALUATION_REPORT.txt', 'w') as f:
    f.write(report_text)
print(f'\n✓ Report saved: COMPREHENSIVE_EVALUATION_REPORT.txt')
print('\n' + report_text)

print('\n' + '='*80)
print('ANALYSIS COMPLETE!')
print('='*80)
print(f'Results: {RESULTS_DIR}')
print(f'Visualizations: {VIS_DIR}')
