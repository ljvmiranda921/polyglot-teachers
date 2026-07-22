"""TEMP (delete later): rebuild generalization_base_models Spearman heatmap
in the website's style. Values are the exact ones from the paper figure.
Chia blue sequential colormap + Helvetica.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "text.usetex": False, "axes.edgecolor": "#1a1a1a",
})
INK = "#1a1a1a"
N = np.nan

# y top->bottom, x left->right (lower-triangular, as in the paper)
ylab = ["Llama 3 8B", "Qwen 3 8B", "Gemma 3 4B", "OLMo 3 7B"]
xlab = ["OLMo 3 7B", "Gemma 3 4B", "Qwen 3 8B", "Llama 3 8B"]
M = np.array([
    [0.63, 0.68, 0.57, 1.00],
    [0.60, 0.65, 1.00, N],
    [0.87, 1.00, N, N],
    [1.00, N, N, N],
])
ann = [
    ["0.63", "0.68*", "0.57", "1.00"],
    ["0.60", "0.65", "1.00", ""],
    ["0.87**", "1.00", "", ""],
    ["1.00", "", "", ""],
]

cmap = LinearSegmentedColormap.from_list("chia_seq", ["#eef1ff", "#254eff"])
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(6.2, 5.8))
ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
for i in range(4):
    for j in range(4):
        if ann[i][j]:
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=15.5,
                    color="white" if M[i, j] > 0.8 else INK)

ax.set_xticks(range(4)); ax.set_xticklabels(xlab, rotation=45, ha="left", fontsize=13.5)
ax.xaxis.set_ticks_position("top")
ax.set_yticks(range(4)); ax.set_yticklabels(ylab, fontsize=13.5)
ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", lw=2.5)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(0.98, 0.16, "**  p < 0.01\n*   p < 0.05", transform=ax.transAxes,
        ha="right", va="top", fontsize=12.5, color="#5a554f")

Path("plot_outputs").mkdir(exist_ok=True)
fig.savefig("plot_outputs/generalization_base_models.pdf", bbox_inches="tight")
fig.savefig("/tmp/prev_generalization.png", dpi=150, bbox_inches="tight")
print("wrote plot_outputs/generalization_base_models.pdf")
