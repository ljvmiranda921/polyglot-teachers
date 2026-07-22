"""TEMP (delete later): rebuild tgl_ablation_filbench_scores in the paper's
layout, recolored to chia (grey hatched baselines, blue recipe bars) + Helvetica.
Values from the website ablation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "text.usetex": False, "axes.edgecolor": "#1a1a1a",
})
BLUE, GREY, INK = "#254eff", "#c9c2b6", "#1a1a1a"
vals = [47.2, 47.7, 48.2, 49.5, 49.7, 51.4, 53.0]
inter = [
    "+Use synthetic\npipeline", "+Better\nteacher", "+Match\nfamily",
    r"+Scale data" "\n" r"(10k$\rightarrow$25k)",
    r"+Scale size" "\n" r"(4B$\rightarrow$12B)",
    r"+Scale size" "\n" r"(12B$\rightarrow$27B)",
]

fig, ax = plt.subplots(figsize=(10, 4.6))
xs = list(range(7))
for i, v in enumerate(vals):
    base = i < 2
    ax.bar(i, v, width=0.72, zorder=3, edgecolor=INK, linewidth=1.3,
           facecolor="white" if base else BLUE,
           hatch="////" if base else None,
           color=None)
    if base:
        ax.patches[-1].set_facecolor("#d9d5cd")
    ax.text(i, v + 1.4, f"{v:.1f}", ha="center", va="bottom", fontsize=14, color=INK)

# intervention arrows (bar top to bar top) with labels near the top
for i in range(6):
    ax.annotate("", xy=(i + 1, 61), xytext=(i, 61),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                connectionstyle="arc3,rad=-0.4"), zorder=4)
    ax.text(i + 0.5, 66, inter[i], ha="center", va="bottom", fontsize=11, color=INK)

ax.set_ylim(0, 84)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel("FilBench Score")
ax.set_xticks(xs)
ax.set_xticklabels(["None", "GPT-4o", "Aya Exp", "", "", "", ""])
ax.set_xlabel("Teacher Model", labelpad=34)
ax.grid(True, axis="y", color="#dddddd", lw=0.8, linestyle="--")
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# "Gemma 3 27B" bracket under the last four bars
yb = -7
ax.plot([2.72, 6.28], [yb, yb], color=INK, lw=1.3, clip_on=False)
ax.plot([2.72, 2.72], [yb, yb + 3], color=INK, lw=1.3, clip_on=False)
ax.plot([6.28, 6.28], [yb, yb + 3], color=INK, lw=1.3, clip_on=False)
ax.text(4.5, yb - 3, "Gemma 3 27B", ha="center", va="top", fontsize=14, clip_on=False)

Path("plot_outputs").mkdir(exist_ok=True)
fig.savefig("plot_outputs/tgl_ablation_filbench_scores.pdf", bbox_inches="tight")
fig.savefig("/tmp/prev_ablation.png", dpi=150, bbox_inches="tight")
print("wrote plot_outputs/tgl_ablation_filbench_scores.pdf")
