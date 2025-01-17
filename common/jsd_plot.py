import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['AFFIA3k', 'UFFIA', 'MRS-FFIA']
frame_mix = [0.004156895340062105, 0.0020401528693744305, 0.001156933605670929]
diff_res = [0.6471150288215051, 0.6400290883590127, 0.6391609464922259]
spec_augment = [0.005746503288929279, 0.007153708117831254, 0.008953749833087768]
spec_mix = [0.009501505858049942, 0.01025077953682255, 0.011916801033000793]

# Bar width and positions
x = np.arange(len(datasets))
width = 0.2

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with colors
bars_frame_mix = ax.bar(x - 1.5 * width, frame_mix, width, label='FrameMixer (ours)', color='blue')
bars_diff_res = ax.bar(x - 0.5 * width, [0.017] * len(datasets), width, label='DiffRes', color='red')
bars_spec_mix = ax.bar(x + 0.5 * width, spec_mix, width, label='SpecMix', color='green')
bars_spec_augment = ax.bar(x + 1.5 * width, spec_augment, width, label='SpecAugment', color='orange')

# Annotate actual DiffRes values above the bars
for i, val in enumerate(diff_res):
    ax.text(x[i] - 0.5 * width, 0.017, f"{val:.5f}", ha='center', va='bottom', fontsize=9)

# Labels and title
ax.set_xlabel('Datasets', fontsize=12)
ax.set_ylabel('Jensen-Shannon Divergence', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 0.017)

# Legend
ax.legend()

# Grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save plot as a figure
plt.tight_layout()
plt.savefig("figures/jsd_plot.png", dpi=300)
