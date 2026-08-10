"""TEMP (delete later): rebuild data_scale_effect in the website's style.
Values from the website SFT chart. One blue color, markers per language."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "text.usetex": False, "axes.edgecolor": "#1a1a1a",
})
BLUE, INK = "#254eff", "#1a1a1a"
x = [1000, 5000, 10000, 25000, 50000]
S = {
    "German": ("s", "-", [0.496, 0.553, 0.639, 0.642, 0.643]),
    "Arabic": ("o", "--", [0.441, 0.541, 0.617, 0.615, 0.628]),
    "Indonesian": ("^", "-.", [0.451, 0.484, 0.557, 0.581, 0.597]),
}
fig, ax = plt.subplots(figsize=(6.4, 5.0))
for lang, (mk, ls, y) in S.items():
    ax.plot(x, y, ls, color=BLUE, lw=2, marker=mk, ms=9,
            markeredgecolor=INK, markeredgewidth=1, label=lang)
ax.set_xscale("log")
ax.set_xticks([1000, 10000]); ax.set_xticklabels(["1k", "10k"])
ax.set_xlabel("Number of SFT examples (log)")
ax.set_ylabel("Avg. multilingual performance")
ax.legend(loc="lower right", frameon=False)
Path("plot_outputs").mkdir(exist_ok=True)
fig.savefig("plot_outputs/data_scale_effect.pdf", bbox_inches="tight")
fig.savefig("/tmp/prev_data_scale.png", dpi=150, bbox_inches="tight")
print("wrote plot_outputs/data_scale_effect.pdf")
