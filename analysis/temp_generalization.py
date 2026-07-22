"""TEMP (delete later): base_model correlation heatmap. Copied verbatim from
analysis/base_model_effect.py:plot_correlation_heatmap, values hardcoded from
the paper figure, colormap swapped to chia blue.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analysis.utils.plot_theme import PLOT_PARAMS, OUTPUT_DIR

plt.rcParams.update(PLOT_PARAMS)

order = ["OLMo 3 7B", "Gemma 3 4B", "Qwen 3 8B", "Llama 3 8B"]
corr = pd.DataFrame([
    [1.00, 0.87, 0.60, 0.63],
    [0.87, 1.00, 0.65, 0.68],
    [0.60, 0.65, 1.00, 0.57],
    [0.63, 0.68, 0.57, 1.00],
], index=order, columns=order)
pval = pd.DataFrame([
    [0.0, 0.005, 0.20, 0.20],
    [0.005, 0.0, 0.20, 0.03],
    [0.20, 0.20, 0.0, 0.20],
    [0.20, 0.03, 0.20, 0.0],
], index=order, columns=order)

base_models = order
annot = corr.copy().astype(str)
for i in range(4):
    for j in range(4):
        v, p = corr.iloc[i, j], pval.iloc[i, j]
        if i == j:
            annot.iloc[i, j] = f"{v:.2f}"
        elif p < 0.01:
            annot.iloc[i, j] = f"{v:.2f}**"
        elif p < 0.05:
            annot.iloc[i, j] = f"{v:.2f}*"
        else:
            annot.iloc[i, j] = f"{v:.2f}"

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
# chia blue sequential (was white -> light_blue -> cambridge_blue teal)
cmap = LinearSegmentedColormap.from_list("chia_corr", ["#FFFFFF", "#CDD6FF", "#254EFF"])

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, mask=mask, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1,
            square=True, cbar=False, ax=ax, xticklabels=base_models,
            yticklabels=base_models, annot_kws={"fontsize": 24})
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left", va="bottom")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", va="center")
ax.text(0.95, 0.35, "** :$p < 0.01$\n* :$p < 0.05$", transform=ax.transAxes,
        fontsize=25, ha="right", va="top")

plt.tight_layout()
Path("plot_outputs").mkdir(exist_ok=True)
plt.savefig("plot_outputs/generalization_base_models.pdf", format="pdf", bbox_inches="tight")
plt.savefig("/tmp/prev_generalization.png", dpi=130, bbox_inches="tight")
print("wrote plot_outputs/generalization_base_models.pdf")
