# import matplotlib.pyplot as plt
# import numpy as np

# # Colors dictionary
# colors = {
#     'FrameMixer': 'lightcoral',  # Light red
#     'DiffRes': 'lightblue',      # Light blue
#     'SpecMix': 'lightgreen',     # Light green
#     'SpecAugment': 'lightsalmon' # Light orange
# }

# # Data
# datasets = ['AFFIA3k', 'AV-FFIA', 'MRS-FFIA']
# frame_mix = [0.004156895340062105, 0.0020401528693744305, 0.001156933605670929]
# diff_res = [0.6471150288215051, 0.6400290883590127, 0.6391609464922259]
# spec_augment = [0.005746503288929279, 0.007153708117831254, 0.008953749833087768]
# spec_mix = [0.009501505858049942, 0.01025077953682255, 0.011916801033000793]

# # Bar width and positions
# x = np.arange(len(datasets))
# width = 0.2

# # Create figure
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot bars with specified colors and hatch patterns
# bars_frame_mix = ax.bar(x - 1.5 * width, frame_mix, width, label='FrameMixer (Ours)',
#                         color=colors['FrameMixer'], hatch='///')
# bars_diff_res = ax.bar(x - 0.5 * width, [0.017] * len(datasets), width, label='DiffRes',
#                        color=colors['DiffRes'], hatch='xxx')
# bars_spec_mix = ax.bar(x + 0.5 * width, spec_mix, width, label='SpecMix',
#                        color=colors['SpecMix'], hatch='---')
# bars_spec_augment = ax.bar(x + 1.5 * width, spec_augment, width, label='SpecAugment',
#                            color=colors['SpecAugment'], hatch='o')

# # Annotate actual DiffRes values above the bars
# for i, val in enumerate(diff_res):
#     ax.text(x[i] - 0.5 * width, 0.017, f"{val:.5f}", ha='center', va='bottom', fontsize=9)

# # Labels and title
# ax.set_xlabel('Datasets', fontsize=18)
# ax.set_ylabel('Jensen-Shannon Divergence', fontsize=18)
# ax.set_xticks(x)
# ax.set_xticklabels(datasets, fontsize=15)
# ax.set_ylim(0, 0.017)

# # Legend outside the plot at the top with font size 13
# ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=4, fontsize=13)

# # Grid for better readability
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Save and show the figure
# plt.tight_layout()
# plt.savefig("figures/jsd_plot.png", dpi=300)
# plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['AFFIA3k', 'AV-FFIA', 'MRS-FFIA']
frame_mix = [0.004156895340062105, 0.0020401528693744305, 0.001156933605670929]
diff_res = [0.6471150288215051, 0.6400290883590127, 0.6391609464922259]
spec_augment = [0.005746503288929279, 0.007153708117831254, 0.008953749833087768]
spec_mix = [0.009501505858049942, 0.01025077953682255, 0.011916801033000793]

# Filter data for 'AFFIA3k'
index = datasets.index('AFFIA3k')
frame_mix_filtered = [frame_mix[index]]
diff_res_filtered = [diff_res[index]]
spec_augment_filtered = [spec_augment[index]]
spec_mix_filtered = [spec_mix[index]]

# Bar width and positions
x = np.arange(1)  # Only one dataset now
width = 0.2

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Define light shades of colors
colors = {
    'FrameMixer': 'lightcoral',  # Light red
    'DiffRes': 'lightblue',       # Light blue
    'SpecMix': 'lightgreen',      # Light green
    'SpecAugment': 'lightsalmon'  # Light orange
}

# Plot bars using light shades of colors
bars_frame_mix = ax.bar(x - 1.5 * width, frame_mix_filtered, width, label='FrameMixer (Ours)',
                        color=colors['FrameMixer'], hatch='///')
bars_diff_res = ax.bar(x - 0.5 * width, [0.017], width, label='DiffRes',
                       color=colors['DiffRes'], hatch='xxx')
bars_spec_mix = ax.bar(x + 0.5 * width, spec_mix_filtered, width, label='SpecMix',
                       color=colors['SpecMix'], hatch='---')
bars_spec_augment = ax.bar(x + 1.5 * width, spec_augment_filtered, width, label='SpecAugment',
                           color=colors['SpecAugment'], hatch='o')

# Annotate actual DiffRes values above the bars
ax.text(x - 0.5 * width, 0.017, f"{diff_res_filtered[0]:.5f}", ha='center', va='bottom', fontsize=9)

# Labels and title
ax.set_xlabel('Datasets', fontsize=12)  # Reduced font size
ax.set_ylabel('Jensen-Shannon Divergence', fontsize=12)  # Reduced font size
ax.set_xticks(x)
ax.set_xticklabels(['AFFIA3k'], fontsize=12)  # Reduced font size
ax.set_ylim(0, 0.017)

# Legend inside the plot
ax.legend(loc='upper right', fontsize=10)  # Reduced font size and moved inside

# Grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show the figure
plt.tight_layout()
plt.savefig("figures/jsd_plot_affia3k.png", dpi=300)
plt.show()