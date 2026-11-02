"""TEMP (delete later): rebuild pca_loading_factors in the website's style.

Loading matrix is the exact one from the paper (read off the PDF text layer,
also hardcoded on the website). Chia diverging colormap + Helvetica.
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

feats = ["Distinct\nPrompts", "Distinct\nResponses", "Perplexity",
         "Rubric Score\n(M-Prometheus)", "Avg. Prompt\nLength", "Avg. Response\nLength"]
pcs = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
L = np.array([
    [0.073, 0.654, 0.008, 0.744, 0.012, -0.117],
    [0.579, -0.098, -0.017, 0.111, -0.660, 0.456],
    [-0.578, -0.037, 0.017, 0.211, 0.075, 0.784],
    [0.514, -0.237, 0.354, 0.182, 0.678, 0.247],
    [-0.079, 0.388, 0.838, -0.332, -0.171, 0.048],
    [-0.234, -0.596, 0.415, 0.497, -0.265, -0.318],
])

# chia diverging: orange -> panel -> blue, centered at 0
cmap = LinearSegmentedColormap.from_list("chia_div", ["#C96A2E", "#f4f2ee", "#254eff"])

fig, ax = plt.subplots(figsize=(7.4, 6.8))
im = ax.imshow(L, cmap=cmap, vmin=-0.85, vmax=0.85, aspect="equal")
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        v = L[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=14,
                color="white" if abs(v) > 0.5 else INK)

ax.set_xticks(range(6)); ax.set_xticklabels(pcs, fontsize=15)
ax.set_yticks(range(6)); ax.set_yticklabels(feats, fontsize=13.5)
# white gaps between cells
ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.grid(which="minor", color="white", lw=2.5)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.13,
                    ticks=[-0.5, 0.0, 0.5])
cbar.set_label("Loading Strength", fontsize=15)
cbar.ax.tick_params(labelsize=13)
cbar.outline.set_edgecolor(INK)

Path("plot_outputs").mkdir(exist_ok=True)
fig.savefig("plot_outputs/pca_loading_factors.pdf", bbox_inches="tight")
fig.savefig("/tmp/prev_pca_loadings.png", dpi=150, bbox_inches="tight")
print("wrote plot_outputs/pca_loading_factors.pdf")
