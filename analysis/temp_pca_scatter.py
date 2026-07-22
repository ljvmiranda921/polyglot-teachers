"""TEMP (delete later): rebuild pca_predicted_vs_actual in the website's style.

Reuses the 58 points already hardcoded on the website (analysis/_tempdata/
pca_points.json) since the committed data can't reproduce the paper figure.
Website colors + Helvetica so the paper figure matches the site.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "text.usetex": False,
    "axes.edgecolor": "#1a1a1a",
})

INK, SLATE = "#1a1a1a", "#545e73"
# same per-language colors + marker shapes as the website scatter
STYLE = {
    "Arabic": ("#7fd8c9", "o"), "Czech": ("#fd8254", "s"), "German": ("#cdb4e8", "D"),
    "Spanish": ("#29337a", "^"), "Indonesian": ("#ffb81c", "P"), "Japanese": ("#545e73", "X"),
}

pts = json.loads(Path("analysis/_tempdata/pca_points.json").read_text())

fig, ax = plt.subplots(figsize=(6.2, 6.2))
# box-diagonal reference line (matches the website)
ax.plot([0.30, 0.40], [0.30, 0.50], "--", color=INK, lw=1.6, zorder=1)
for lang, (color, marker) in STYLE.items():
    xs = [p["x"] for p in pts if p["l"] == lang]
    ys = [p["y"] for p in pts if p["l"] == lang]
    ax.scatter(xs, ys, s=110, c=color, marker=marker, edgecolors=INK,
               linewidths=1.0, alpha=0.95, label=lang, zorder=3)

ax.set_xlim(0.295, 0.405); ax.set_xticks([0.30, 0.35, 0.40])
ax.set_ylim(0.295, 0.505); ax.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50])
ax.set_xlabel("Actual Benchmark Score")
ax.set_ylabel("Predicted Benchmark Score")
ax.grid(True, color="#eeeeee", lw=0.8)
ax.set_axisbelow(True)
ax.text(0.97, 0.05, "R² = 0.664\nRMSE = 0.440", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK, lw=1.0))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

Path("plot_outputs").mkdir(exist_ok=True)
fig.savefig("plot_outputs/pca_predicted_vs_actual_linear.pdf", bbox_inches="tight")
fig.savefig("/tmp/prev_pca_scatter.png", dpi=150, bbox_inches="tight")
print("wrote plot_outputs/pca_predicted_vs_actual_linear.pdf")
