#!/usr/bin/env python3
"""Generate the E3 injection-curve grouped-bar chart.

Plots five classification metrics at seven Cholec80 injection ratios
(0 %–100 %), averaged over three runs with error bars.

Outputs:
    injection_curve.pdf
    injection_curve.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8,
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
# Cholec80 injection fractions used in Experiment 3.
fractions = ["0%", "10%", "25%", "50%", "65%", "80%", "100%"]

# Per-run results (three independent runs) at each injection fraction.
seed_data = [
    # Run 1
    {
        "Per-video\nAccuracy": [0.8932, 0.9012, 0.8964, 0.9012, 0.8921, 0.8934, 0.8908],
        "Frame\nAccuracy":     [0.9123, 0.9179, 0.9134, 0.9139, 0.9076, 0.9103, 0.9085],
        "F1\n(macro)":         [0.7085, 0.7648, 0.7535, 0.8085, 0.7452, 0.7284, 0.7237],
        "Precision\n(macro)":  [0.8155, 0.8780, 0.8324, 0.9230, 0.8276, 0.7495, 0.7614],
        "Recall\n(macro)":     [0.6778, 0.7325, 0.7299, 0.7646, 0.7208, 0.7124, 0.7049],
    },
    # Run 2
    {
        "Per-video\nAccuracy": [0.8437, 0.8289, 0.8146, 0.8171, 0.8272, 0.8178, 0.8354],
        "Frame\nAccuracy":     [0.8460, 0.8376, 0.8217, 0.8268, 0.8481, 0.8151, 0.8320],
        "F1\n(macro)":         [0.6969, 0.6682, 0.7040, 0.6480, 0.6635, 0.6777, 0.7295],
        "Precision\n(macro)":  [0.7411, 0.7168, 0.7849, 0.6966, 0.6935, 0.7624, 0.8055],
        "Recall\n(macro)":     [0.6736, 0.6441, 0.6653, 0.6244, 0.6455, 0.6443, 0.6917],
    },
    # Run 3
    {
        "Per-video\nAccuracy": [0.7465, 0.8188, 0.8170, 0.8027, 0.8259, 0.8248, 0.8173],
        "Frame\nAccuracy":     [0.7435, 0.8198, 0.8133, 0.8171, 0.8319, 0.8267, 0.8174],
        "F1\n(macro)":         [0.5462, 0.6149, 0.6574, 0.6569, 0.6606, 0.6621, 0.6539],
        "Precision\n(macro)":  [0.6563, 0.6993, 0.7027, 0.6808, 0.6803, 0.6936, 0.6872],
        "Recall\n(macro)":     [0.5321, 0.5890, 0.6381, 0.6445, 0.6531, 0.6460, 0.6391],
    },
]

# ── Aggregate across runs ────────────────────────────────────────────────────
metrics = list(seed_data[0].keys())
data_mean = {}
data_std = {}
for metric in metrics:
    vals = np.array([seed[metric] for seed in seed_data], dtype=float)
    data_mean[metric] = np.mean(vals, axis=0)
    data_std[metric] = np.std(vals, axis=0, ddof=1)
n_metrics = len(metrics)
n_fracs = len(fractions)

# ── Print summary table ──────────────────────────────────────────────────────
col_w_metric = 20
col_w_num    = 11
header = (
    f"{'Metric':<{col_w_metric}}"
    f"{'Fraction':<{10}}"
    f"{'Mean':>{col_w_num}}"
    f"{'Std Dev':>{col_w_num}}"
)
print("\nSummary table (mean and std dev)")
print(header)
print("-" * len(header))
for metric in metrics:
    metric_name = metric.replace("\n", " ")
    for i, frac in enumerate(fractions):
        print(
            f"{metric_name:<{col_w_metric}}"
            f"{frac:<{10}}"
            f"{data_mean[metric][i]:>{col_w_num}.4f}"
            f"{data_std[metric][i]:>{col_w_num}.6f}"
        )
    print("-" * len(header))

# ── Colour palette (one per fraction) ────────────────────────────────────────
colors  = ["#4f83cc", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#b07aa1", "#9c755f"]
hatches = ["", "", "", "", "", "", ""]

# ── Build figure ─────────────────────────────────────────────────────────────
x = np.arange(n_metrics)
total_width = 0.75
bar_w = total_width / n_fracs

fig, ax = plt.subplots(figsize=(7.16, 2.4))

for i, (frac, color, hatch) in enumerate(zip(fractions, colors, hatches)):
    vals = [data_mean[m][i] for m in metrics]
    errs = [data_std[m][i] for m in metrics]
    offset = (i - (n_fracs - 1) / 2) * bar_w
    bars = ax.bar(
        x + offset, vals, bar_w * 0.72,
        label=f"p = {frac}",
        color=color, alpha=0.82,
        edgecolor="white", linewidth=0.4,
        hatch=hatch, zorder=3,
        yerr=errs, ecolor="#777777", capsize=1.5,
        error_kw={"elinewidth": 0.55, "alpha": 0.7},
    )


ax.set_xticks(x)
ax.set_xticklabels(metrics, ha="center")
ax.set_ylabel("Score")
ax.set_ylim(0.55, 0.92)
ax.set_xlim(-0.5, n_metrics - 0.5)

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

legend = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.18),
    ncol=n_fracs, frameon=False, handlelength=1.4, handletextpad=0.4,
    columnspacing=1.2,
)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig("injection_curve.pdf")
fig.savefig("injection_curve.png", dpi=300)
print("Saved  injection_curve.pdf")
print("Saved  injection_curve.png")
