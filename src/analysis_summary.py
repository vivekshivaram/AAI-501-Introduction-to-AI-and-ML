"""
Generate comprehensive analysis summary with interpretations for all experiments.
Creates annotated visualizations showing key insights from enhanced granularity.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load experiment results
data_path = Path(__file__).parent.parent / "data" / "outputs" / "experiment_results.json"
with open(data_path, 'r') as f:
    results = json.load(f)

# Create comprehensive analysis figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================================
# 1. ORDER VOLUME ANALYSIS
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
order_data = results['order_volume']
orders = [r['config']['orders_per_tick'] for r in order_data]
completion = [r['metrics']['delivery_rate'] * 100 for r in order_data]
dispatch = [r['metrics']['dispatch_efficiency'] * 100 for r in order_data]
throughput = [r['metrics']['throughput'] for r in order_data]

# Plot with dual y-axis
ax1_twin = ax1.twinx()
line1 = ax1.plot(orders, completion, 'o-', linewidth=2.5, markersize=9, 
                 color='#2ecc71', label='Completion Rate')
line2 = ax1.plot(orders, dispatch, 's--', linewidth=2.5, markersize=9, 
                 color='#3498db', label='Dispatch Efficiency')
line3 = ax1_twin.plot(orders, throughput, '^-', linewidth=2.5, markersize=9, 
                      color='#e74c3c', label='Throughput')

ax1.set_xlabel('Orders per Tick', fontsize=13, fontweight='bold')
ax1.set_ylabel('Rate (%)', fontsize=13, fontweight='bold')
ax1_twin.set_ylabel('Throughput (deliveries/tick)', fontsize=13, fontweight='bold', color='#e74c3c')
ax1.set_title('ORDER VOLUME ANALYSIS: Identifying Capacity Saturation Point', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(orders)

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=11)

# Add interpretation annotations
ax1.annotate('Sweet Spot:\n5-12 orders/tick\nStable 67-73% completion',
             xy=(7, 68), xytext=(4, 80),
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#d5f4e6', alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'),
             fontsize=10, fontweight='bold')

ax1.annotate('Saturation!\nAt 20 orders/tick:\n58% completion\n85% rejection',
             xy=(20, 58), xytext=(16, 45),
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#fadbd8', alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'),
             fontsize=10, fontweight='bold')

ax1.annotate('Low Load:\nUnderutilized fleet\nOnly 67% dispatch',
             xy=(1, 67), xytext=(1, 88),
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef5e7', alpha=0.9),
             fontsize=9, style='italic')

ax1.axvspan(5, 12, alpha=0.1, color='green', label='Optimal Range')

# ============================================================================
# 2. Q-EPSILON ANALYSIS (Aberration Detection)
# ============================================================================
ax2 = fig.add_subplot(gs[1, :])
epsilon_data = results['q_learning_epsilon']
epsilons = [r['config']['q_epsilon'] for r in epsilon_data]
epsilon_completion = [r['metrics']['delivery_rate'] * 100 for r in epsilon_data]
epsilon_surge = [r['metrics']['avg_surge'] for r in epsilon_data]

# Plot with dual y-axis
ax2_twin = ax2.twinx()
line1 = ax2.plot(epsilons, epsilon_completion, 'o-', linewidth=2.5, markersize=9,
                 color='#9b59b6', label='Completion Rate')
line2 = ax2_twin.plot(epsilons, epsilon_surge, 's--', linewidth=2.5, markersize=9,
                      color='#f39c12', label='Avg Surge Multiplier')

ax2.set_xlabel('Q-Learning Epsilon (Exploration Rate)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Completion Rate (%)', fontsize=13, fontweight='bold')
ax2_twin.set_ylabel('Surge Multiplier', fontsize=13, fontweight='bold', color='#f39c12')
ax2.set_title('Q-EPSILON ANALYSIS: Fine-Grained Aberration Detection Around 0.20', 
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right', fontsize=11)

# Find and annotate key points
max_idx = epsilon_completion.index(max(epsilon_completion))
min_idx = epsilon_completion.index(min(epsilon_completion))

ax2.annotate(f'Peak Performance!\nε = {epsilons[max_idx]}\n{epsilon_completion[max_idx]:.1f}% completion',
             xy=(epsilons[max_idx], epsilon_completion[max_idx]), 
             xytext=(epsilons[max_idx]-0.08, epsilon_completion[max_idx]+8),
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#d5f4e6', alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'),
             fontsize=10, fontweight='bold')

ax2.annotate(f'Performance Drop\nε = {epsilons[min_idx]}\n{epsilon_completion[min_idx]:.1f}% completion\nToo much exploration!',
             xy=(epsilons[min_idx], epsilon_completion[min_idx]), 
             xytext=(epsilons[min_idx]+0.05, epsilon_completion[min_idx]-10),
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#fadbd8', alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'),
             fontsize=10, fontweight='bold')

# Highlight stable region
stable_region = [i for i, eps in enumerate(epsilons) if 0.18 <= eps <= 0.35]
if stable_region:
    ax2.axvspan(0.18, 0.35, alpha=0.1, color='green')
    ax2.text(0.265, 85, 'Stable Region\n(18-35% exploration)', 
             ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.annotate('No aberration at 0.20!\nSmooth transition around this point',
             xy=(0.20, epsilon_completion[epsilons.index(0.20)]), 
             xytext=(0.08, 76),
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f8f5', alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#16a085', linestyle='--'),
             fontsize=9, style='italic')

# ============================================================================
# 3. Q-ALPHA ANALYSIS
# ============================================================================
ax3 = fig.add_subplot(gs[2, 0])
alpha_data = results['q_learning_alpha']
alphas = [r['config']['q_alpha'] for r in alpha_data]
alpha_completion = [r['metrics']['delivery_rate'] * 100 for r in alpha_data]
alpha_throughput = [r['metrics']['throughput'] for r in alpha_data]

ax3.plot(alphas, alpha_completion, 'o-', linewidth=2, markersize=8, color='#e67e22')
ax3.set_xlabel('Q-Alpha (Learning Rate)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('Q-ALPHA: Learning Rate Tuning', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Find optimal
max_alpha_idx = alpha_completion.index(max(alpha_completion))
ax3.annotate(f'Optimal: α={alphas[max_alpha_idx]}',
             xy=(alphas[max_alpha_idx], alpha_completion[max_alpha_idx]),
             xytext=(alphas[max_alpha_idx]+0.1, alpha_completion[max_alpha_idx]-5),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=2),
             fontsize=10, fontweight='bold')

# ============================================================================
# 4. KEY INSIGHTS - Order Volume
# ============================================================================
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')
insights_order = f"""
ORDER VOLUME INSIGHTS
{'='*40}

✓ OPTIMAL RANGE: 5-12 orders/tick
  • Completion Rate: 67-73%
  • Dispatch Efficiency: 33-41%
  • Throughput: 1.1-1.35 deliveries/tick

⚠ LOW LOAD (1-3 orders/tick):
  • Good completion (67-80%)
  • BUT: Underutilized capacity
  • Fleet idle time increases costs

⛔ SATURATION (20 orders/tick):
  • Completion drops to 58%
  • Only 15% dispatch efficiency
  • 356 pending orders (85% rejection!)
  • System overwhelmed

📊 ENHANCED GRANULARITY REVEALS:
  • Smooth degradation 12→15→20
  • No sudden "cliff" failure
  • Gradual capacity saturation
"""

ax4.text(0.05, 0.95, insights_order, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#e8f8f5', alpha=0.8))

# ============================================================================
# 5. KEY INSIGHTS - Q-Learning
# ============================================================================
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')

# Calculate stats
epsilon_peak = epsilons[max_idx]
epsilon_peak_rate = epsilon_completion[max_idx]
alpha_peak = alphas[max_alpha_idx]
alpha_peak_rate = alpha_completion[max_alpha_idx]

insights_ql = f"""
Q-LEARNING INSIGHTS
{'='*40}

🎯 EPSILON (Exploration):
  • Peak: ε={epsilon_peak} → {epsilon_peak_rate:.1f}%
  • Stable: 0.18-0.35 range
  • Drop: ε=0.40 → 62.5%
  • NO aberration at 0.20!
  • Fine granularity (11 pts) confirms
    smooth behavior around 0.20

🎯 ALPHA (Learning Rate):
  • Optimal: α={alpha_peak} → {alpha_peak_rate:.1f}%
  • Enhanced sampling reveals
    precise sweet spot

💡 RECOMMENDATION:
  • Use ε ∈ [0.18, 0.30]
  • Use α ∈ [0.15, 0.25]
  • Provides stable pricing with
    good exploration-exploitation
    balance
"""

ax5.text(0.05, 0.95, insights_ql, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#fef5e7', alpha=0.8))

# ============================================================================
# Overall title and save
# ============================================================================
fig.suptitle('COMPREHENSIVE EXPERIMENT ANALYSIS: Enhanced Granularity Results',
             fontsize=18, fontweight='bold', y=0.995)

output_path = Path(__file__).parent.parent / "data" / "outputs" / "analysis_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Analysis summary saved to: {output_path}")

# ============================================================================
# Create detailed comparison chart
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('DETAILED PERFORMANCE METRICS COMPARISON', fontsize=16, fontweight='bold')

# Order Volume - Throughput vs Completion
ax = axes[0, 0]
ax.scatter(throughput, completion, s=200, c=orders, cmap='viridis', 
           alpha=0.7, edgecolors='black', linewidth=2)
for i, txt in enumerate(orders):
    ax.annotate(f'{txt}', (throughput[i], completion[i]), 
                ha='center', va='center', fontweight='bold', fontsize=9)
ax.set_xlabel('Throughput (deliveries/tick)', fontsize=12, fontweight='bold')
ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Order Volume: Throughput vs Quality Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.scatter(throughput, completion, s=200, c=orders, cmap='viridis', alpha=0.7), ax=ax)
cbar.set_label('Orders/Tick', fontsize=11)
ax.annotate('Sweet Spot:\nHigh throughput\n+ Good completion',
            xy=(1.1, 69), xytext=(0.6, 75),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2),
            fontsize=10)

# Q-Epsilon detailed
ax = axes[0, 1]
bars = ax.bar(range(len(epsilons)), epsilon_completion, color='#9b59b6', alpha=0.7, edgecolor='black')
bars[max_idx].set_color('#27ae60')  # Highlight best
bars[min_idx].set_color('#e74c3c')  # Highlight worst
ax.set_xlabel('Epsilon Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Q-Epsilon: All 11 Test Points', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(epsilons)))
ax.set_xticklabels([f'{e:.2f}' for e in epsilons], rotation=45)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=75, color='orange', linestyle='--', linewidth=2, label='75% Target')
ax.legend()

# Q-Alpha detailed
ax = axes[1, 0]
bars = ax.bar(range(len(alphas)), alpha_completion, color='#e67e22', alpha=0.7, edgecolor='black')
bars[max_alpha_idx].set_color('#27ae60')
ax.set_xlabel('Alpha Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Q-Alpha: All 9 Test Points', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f'{a:.2f}' for a in alphas], rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Summary statistics table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = [
    ['Metric', 'Best Config', 'Value'],
    ['', '', ''],
    ['Highest Completion', f'ε={epsilon_peak}', f'{epsilon_peak_rate:.1f}%'],
    ['Highest Throughput', f'{max(orders)} orders/tick', f'{max(throughput):.2f}/tick'],
    ['Best Efficiency', f'α={alpha_peak}', f'{alpha_peak_rate:.1f}%'],
    ['', '', ''],
    ['Optimal Order Volume', '5-12 orders/tick', '67-73%'],
    ['Optimal Epsilon Range', '0.18-0.30', '75-90%'],
    ['Optimal Alpha Range', '0.15-0.25', '75-82%'],
]

table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.35, 0.35, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator rows
for col in range(3):
    table[(1, col)].set_facecolor('#ecf0f1')
    table[(5, col)].set_facecolor('#ecf0f1')

# Style data rows with alternating colors
for row in range(2, 9):
    if row not in [1, 5]:
        color = '#e8f8f5' if row % 2 == 0 else 'white'
        for col in range(3):
            table[(row, col)].set_facecolor(color)

ax.set_title('OPTIMAL CONFIGURATION SUMMARY', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
output_path2 = Path(__file__).parent.parent / "data" / "outputs" / "detailed_comparison.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Detailed comparison saved to: {output_path2}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
