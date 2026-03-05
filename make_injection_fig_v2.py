import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
})

# ──────────────────────────────────────────────────────────────────────────────
# Updated data from your logs (0.00 / 0.10 / 0.25 / 0.50 / 0.65 / 0.80 / 1.00)
# Classes: C0 Prep · C1 GB Retract · C2 Calot · C3 Clip/Cut · C4 GB Dissect
# ──────────────────────────────────────────────────────────────────────────────

ratios = [0, 10, 25, 50, 65, 80, 100]
ratio_labels = ['0%\n(SITL only)', '10%', '25%', '50%', '65%', '80%', '100%']

# ✅ Requested: divide all SITL training samples by 2
SITL_SCALE = 1

# SITL-only baseline train split (pre-injection) — constant across runs
sitl_only_full = {0: 9887, 1: 3351, 2: 203360, 3: 31855, 4: 231868}  # total 480,321
sitl_only = {k: v * SITL_SCALE for k, v in sitl_only_full.items()}
sitl_total = sum(sitl_only.values())

# Injected Cholec80 frames (added into train) per ratio (unchanged)
cholec_injected = {
    0:   {0:    0, 1:    0, 2:     0, 3:    0, 4:     0},   # 0
    10:  {0:  703, 1:  198, 2:  5118, 3:  753, 4:  2630},   #  9,402
    25:  {0: 1334, 1:  521, 2: 15534, 3: 2237, 4:  5672},   # 25,298
    50:  {0: 1977, 1: 1777, 2: 22277, 3: 3783, 4: 10242},   # 40,056
    65:  {0: 2928, 1: 2150, 2: 27475, 3: 4876, 4: 14130},   # 51,559
    80:  {0: 3430, 1: 2606, 2: 30549, 3: 5713, 4: 17634},   # 59,932
    100: {0: 3758, 1: 3277, 2: 36886, 3: 7329, 4: 24119},   # 75,369
}
cholec_totals = [sum(cholec_injected[r].values()) for r in ratios]

# Recompute updated train counts BEFORE upsampling under the new SITL scaling:
# updated[r] = (scaled SITL) + (injected Cholec80)
updated = {}
for r in ratios:
    updated[r] = {ci: sitl_only[ci] + cholec_injected[r][ci] for ci in range(5)}

# Recompute upsampling target dynamically (median of class counts) per ratio,
# then upsample C0/C1 to that target (matches your pipeline logic).
train_final = {}
upsample_target = {}
for r in ratios:
    vals = np.array(list(updated[r].values()), dtype=float)
    target = float(np.median(vals))
    upsample_target[r] = target
    train_final[r] = dict(updated[r])
    train_final[r][0] = target
    train_final[r][1] = target

# Choose what to plot as "class distribution":
#   True  = distribution the TRAINER sees (after upsampling)
#   False = distribution of unique frames (before upsampling)
USE_UPSAMPLED_FOR_CLASS_PLOT = False

# Left panel option:
#   True  = show final train size stacked as SITL + Cholec + (upsampling duplicates)
#   False = show unique frames stacked as SITL + Cholec (and annotate final)
SHOW_UPSAMPLED_DUPLICATES_ON_LEFT = True

# ── Derived totals ──
unique_totals = [sum(updated[r].values()) for r in ratios]
final_totals  = [sum(train_final[r].values()) for r in ratios]
dup_totals    = [f - u for f, u in zip(final_totals, unique_totals)]  # added via upsampling

# ── Labels/colors ──
class_names = ['Preparation', 'GB Retraction', 'Calot Dissect.', 'Clip./Cutting', 'GB Dissection']
colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw={'width_ratios': [1, 1.7]})

# ── Left panel: training set size composition ──
ax = axes[0]
x = np.arange(len(ratios))
bar_w = 0.55

# SITL base (scaled, constant)
ax.bar(
    x, [sitl_total] * len(ratios), bar_w,
    label=f'SITL (robotic) ×{SITL_SCALE:g}', color='#4e79a7',
    edgecolor='white', linewidth=0.5
)

# Cholec injected (variable)
ax.bar(
    x, cholec_totals, bar_w,
    bottom=[sitl_total] * len(ratios),
    label='Cholec80 injected (lap)', color='#f28e2b',
    edgecolor='white', linewidth=0.5
)

# Optional: upsampling duplicates (extra samples, not new unique frames)
if SHOW_UPSAMPLED_DUPLICATES_ON_LEFT:
    ax.bar(
        x, dup_totals, bar_w,
        bottom=[sitl_total + c for c in cholec_totals],
        label='Upsampling duplicates', color='#bab0ac',
        edgecolor='white', linewidth=0.5
    )

# Annotations
for i, r in enumerate(ratios):
    c = cholec_totals[i]
    d = dup_totals[i]
    f = final_totals[i]
    top = sitl_total + c + (d if SHOW_UPSAMPLED_DUPLICATES_ON_LEFT else 0)

    if c > 0:
        ax.text(i, top + 6000, f'+{c/1000:.1f}k',
                ha='center', va='bottom', fontsize=6.5,
                color='#f28e2b', fontweight='bold')

    ax.text(i, f + 14000, f'{f/1000:.0f}k',
            ha='center', va='bottom', fontsize=6.5, color='#111827')

ax.set_xticks(x)
ax.set_xticklabels(ratio_labels)
ax.set_ylabel('Training samples')
ax.set_xlabel('Cholec80 injection ratio')
ax.set_title('(a) Training set composition')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
ax.set_ylim(0, max(final_totals) * 1.22)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Right panel: class distribution (%) ──
ax2 = axes[1]
dist_source = train_final if USE_UPSAMPLED_FOR_CLASS_PLOT else updated

# C0, C1, C3 are all upsampled to the same median → identical lines.
# Use vertical offsets so their end-of-line labels don't stack on top of each other.
label_y_offsets = {0: 2.5, 1: 0.0, 2: 0.0, 3: -2.5, 4: 0.0}

for ci in range(5):
    pcts = [dist_source[r][ci] / sum(dist_source[r].values()) * 100 for r in ratios]
    ax2.plot(ratios, pcts, 'o-', color=colors[ci], label=class_names[ci],
             markersize=4, linewidth=1.5)
    ax2.text(103, pcts[-1] + label_y_offsets[ci], class_names[ci],
             color=colors[ci], fontsize=7, va='center')

ax2.set_xlabel('Cholec80 injection ratio (%)')
ax2.set_ylabel('Class proportion (%)')
ax2.set_title('(b) Class distribution vs. injection ratio' +
              (' (after upsampling)' if USE_UPSAMPLED_FOR_CLASS_PLOT else ' (unique frames)'))
ax2.set_xticks(ratios)
ax2.set_xticklabels([str(r) for r in ratios])
ax2.set_xlim(-5, 145)
ax2.set_ylim(-1, 55)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout(w_pad=2.0)
plt.savefig('cholec80_injection.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('cholec80_injection.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print("Saved cholec80_injection.pdf and cholec80_injection.png")