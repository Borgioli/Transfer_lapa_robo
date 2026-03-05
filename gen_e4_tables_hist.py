#!/usr/bin/env python3
"""Generate the E4 grouped-bar histogram comparing OmniRAS+SurgeNetLP
vs. OmniRAS-only pretraining across three fine-tuning regimes
(Frozen / Linear / Full) for five classification metrics.

Outputs:
    e4_tables_histogram.pdf
    e4_tables_histogram.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Rectangle

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   7,
    "figure.dpi":        300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.03,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
})

# ── Experiment data ──────────────────────────────────────────────────────────
# Column order: per-video acc, frame acc, F1_m, F1_w, rec_m, rec_w, pre_m, pre_w
# Each array contains three runs (rows) × eight metrics (columns).
table_vii = {
    "Frozen": np.array([
        [0.7681, 0.7872, 0.6373, 0.7823, 0.5910, 0.7872, 0.7626, 0.8019],
        [0.7244, 0.7963, 0.5950, 0.8002, 0.5653, 0.7963, 0.6693, 0.8299],
        [0.7742, 0.7992, 0.6055, 0.7986, 0.5841, 0.7992, 0.6621, 0.8292],
    ], dtype=float),
    "linear layer": np.array([
        [0.7963, 0.7940, 0.6520, 0.7835, 0.6264, 0.7940, 0.7309, 0.8166],
        [0.8099, 0.8971, 0.6902, 0.8988, 0.6847, 0.8971, 0.7051, 0.9108],
        [0.8642, 0.8905, 0.8284, 0.8918, 0.9236, 0.8905, 0.7934, 0.9117],
    ], dtype=float),
    "Full": np.array([
        [0.8430, 0.8508, 0.6776, 0.8438, 0.6597, 0.8508, 0.7145, 0.8546],
        [0.8217, 0.9115, 0.6916, 0.9113, 0.6694, 0.9115, 0.7259, 0.9173],
        [0.8558, 0.8857, 0.8506, 0.8858, 0.8547, 0.8857, 0.8627, 0.9045],
    ], dtype=float),
}

table_viii = {
    "Frozen": np.array([
        [0.7247, 0.7618, 0.5515, 0.7612, 0.5142, 0.7618, 0.6613, 0.7992],
        [0.7742, 0.7833, 0.5624, 0.7759, 0.5330, 0.7833, 0.6624, 0.7920],
        [0.7415, 0.8052, 0.5903, 0.8086, 0.5579, 0.8052, 0.6621, 0.8278],
    ], dtype=float),
    "linear layer": np.array([
        [0.8254, 0.8594, 0.7661, 0.8594, 0.7516, 0.8594, 0.7976, 0.8871],
        [0.8257, 0.8376, 0.6321, 0.8271, 0.6008, 0.8376, 0.7093, 0.8425],
        [0.8058, 0.8865, 0.6638, 0.8873, 0.6690, 0.8865, 0.6700, 0.9002],
    ], dtype=float),
    "Full": np.array([
        [0.8868, 0.9091, 0.8739, 0.9100, 0.9303, 0.9091, 0.8425, 0.9250],
        [0.8221, 0.8201, 0.6487, 0.8090, 0.6371, 0.8201, 0.6955, 0.8358],
        [0.8384, 0.9133, 0.6943, 0.9134, 0.7106, 0.9133, 0.6928, 0.9227],
    ], dtype=float),
}

metrics = [
    "Per-video\nAccuracy",
    "Frame\nAccuracy",
    "F1\n(macro)",
    "Precision\n(macro)",
    "Recall\n(macro)",
]
metric_idx = [0, 1, 2, 6, 4]

REGIME_COLORS = {
    "Frozen":       "#4f83cc",
    "linear layer": "#f28e2b",
    "Full":         "#59a14f",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def mean_std(vals: np.ndarray):
    """Return per-column mean and population std."""
    means = vals.mean(axis=0)
    stds  = np.sqrt(np.mean((vals - means) ** 2, axis=0))
    return means, stds


# ── Build figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.33))
x = np.arange(len(metrics))
order = ["Frozen", "linear layer", "Full"]
y_min, y_max = 0.52, 0.93

# Six bars per metric:
# (SITL+SurgeNet Frozen/linear/Full) + (SITL-only Frozen/linear/Full)
bar_w = 0.11
group_gap = 0.08  # extra space between the two 3-bar groups
offset_shift = group_gap / (2.0 * bar_w)
outer_rect_extension = group_gap / 2.0  # same extension: left for red, right for orange
offsets = np.array([
    -2.5 - offset_shift, -1.5 - offset_shift, -0.5 - offset_shift,
     0.5 + offset_shift,  1.5 + offset_shift,  2.5 + offset_shift,
]) * bar_w

legend_handles = []

# (display key, data dict, bar hatch, rect line-style, rect colour)
setting_order = [
    ("SITL+SurgeNet", table_vii,  "",   "-",  "#2a9d8f"),
    ("SITL-only",     table_viii, "//", "--", "#e76f51"),
]
SETTING_DISPLAY = {
    "SITL+SurgeNet": "OmniRAS+SurgeNetLP",
    "SITL-only":     "OmniRAS-only",
}

setting_regime_means = {}
for setting_idx, (setting_name, setting_data, hatch, line_style, line_color) in enumerate(setting_order):
    setting_regime_means[setting_name] = []
    for i, key in enumerate(order):
        vals = setting_data[key][:, metric_idx]
        means, stds = mean_std(vals)
        setting_regime_means[setting_name].append(means)
        pos = x + offsets[setting_idx * 3 + i]
        bars = ax.bar(
            pos,
            means,
            bar_w * 0.90,
            yerr=stds,
            capsize=1.5,
            color=REGIME_COLORS[key],
            edgecolor="white",
            linewidth=0.4,
            hatch=hatch,
            alpha=0.82,
            error_kw={"elinewidth": 0.55, "ecolor": "#777777", "alpha": 0.7, "capthick": 0.55},
            zorder=3,
        )
        full_setting = SETTING_DISPLAY[setting_name]
        short_key = "Linear" if key == "linear layer" else key
        legend_handles.append((bars[0], f"{full_setting} {short_key}"))

    # Transparent rectangle showing the per-setting mean across regimes.
    regime_means = np.vstack(setting_regime_means[setting_name])  # (3, n_metrics)
    metric_triplet_mean = regime_means.mean(axis=0)
    left_off = offsets[setting_idx * 3] - 0.5 * bar_w * 0.90
    right_off = offsets[setting_idx * 3 + 2] + 0.5 * bar_w * 0.90
    for j, y in enumerate(metric_triplet_mean):
        mid = x[j]
        if setting_idx == 0:
            x0 = x[j] + left_off - outer_rect_extension
            width = mid - x0
        else:
            x0 = mid
            width = (x[j] + right_off + outer_rect_extension) - mid
        height = max(0.001, y - y_min)
        rect = Rectangle(
            (x0, y_min),
            width,
            height,
            facecolor=mcolors.to_rgba(line_color, 0.18),
            edgecolor=line_color,
            linestyle=line_style,
            linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(rect)

ax.set_xticks(x)
ax.set_xticklabels(metrics, ha="center")
ax.set_ylabel("Score")
ax.set_ylim(y_min, y_max)
ax.set_xlim(-0.5, len(metrics) - 0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#999999")
ax.spines["bottom"].set_color("#999999")
ax.tick_params(colors="#555555")
ax.yaxis.label.set_color("#333333")

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax.grid(axis="y", which="major", linewidth=0.3, color="#cccccc", alpha=0.7, zorder=0)
ax.grid(axis="y", which="minor", linewidth=0.15, color="#e0e0e0", alpha=0.5, zorder=0)

handles = [h for h, _ in legend_handles]
labels = [l for _, l in legend_handles]
label_a = "OmniRAS+SurgeNetLP Linear"
label_b = "OmniRAS-only Linear"
if label_a in labels and label_b in labels:
    idx_a, idx_b = labels.index(label_a), labels.index(label_b)
    labels[idx_a], labels[idx_b] = labels[idx_b], labels[idx_a]
    handles[idx_a], handles[idx_b] = handles[idx_b], handles[idx_a]

handles.extend([
    Rectangle((0, 0), 1, 0.25, facecolor=mcolors.to_rgba("#2a9d8f", 0.20), edgecolor="#2a9d8f", linestyle="-", linewidth=1.2),
    Rectangle((0, 0), 1, 0.25, facecolor=mcolors.to_rgba("#e76f51", 0.20), edgecolor="#e76f51", linestyle="--", linewidth=1.2),
])
labels.extend([
    "Mean OmniRAS+SurgeNetLP",
    "Mean OmniRAS-only",
])
# Lock the plot size first, then add the legend on top
fig.tight_layout()
fig.subplots_adjust(top=0.84)  # make room above the axes for the legend

legend_cols = int(np.ceil(len(labels) / 2))  # force two legend rows
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    ncol=legend_cols,
    frameon=False,
    fontsize=6.5,
    handlelength=1.1,
    handletextpad=0.25,
    columnspacing=0.55,
    borderaxespad=0.2,
)
# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig("e4_tables_histogram.pdf")
fig.savefig("e4_tables_histogram.png", dpi=300)
print("Saved  e4_tables_histogram.pdf")
print("Saved  e4_tables_histogram.png")
