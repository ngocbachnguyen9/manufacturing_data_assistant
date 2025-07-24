
import matplotlib.pyplot as plt
import numpy as np

# Data from Findings.md
labels = ['Human', 'LLM']
cost_per_task = [1.154, 0.0004]  # [Human, LLM]
accuracy = [63.3, 97.4]  # [Human, LLM]

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('ROI Comparison: Before and After LLM Implementation', fontsize=20, fontweight='bold')

# --- Bar Chart 1: Cost Per Task Comparison ---
colors_cost = ['#d9534f', '#5cb85c']
bars1 = ax1.bar(labels, cost_per_task, color=colors_cost, width=0.6)
ax1.set_yscale('log') # Use log scale due to huge difference
ax1.set_ylabel('Cost per Task (USD) - Log Scale', fontsize=12, fontweight='bold')
ax1.set_title('Drastic Cost Reduction', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=10)

# Add data labels
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.1, f'${yval:,.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Bar Chart 2: Accuracy Comparison ---
colors_accuracy = ['#d9534f', '#5cb85c']
bars2 = ax2.bar(labels, accuracy, color=colors_accuracy, width=0.6)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Significant Accuracy Improvement', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=10)

# Add data labels
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Final Touches ---
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('roi_comparison.png', dpi=300)

print("✅ ROI comparison chart saved as roi_comparison.png")
