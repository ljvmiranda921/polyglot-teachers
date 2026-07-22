"""TEMP (delete later): Tagalog ablation. Copied from analysis/ablation_results.py,
values hardcoded from the website ablation, bars recolored to chia (white hatched
baselines, blue recipe), y-grid removed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from analysis.utils.plot_theme import PLOT_PARAMS, FONT_SIZES

plt.rcParams.update(PLOT_PARAMS)
BLUE, INK = "#254EFF", "black"


def add_arc_annotation(ax, df, s, e, text, y_offset=2.2, text_y_offset=3.4, text_x_offset=0, rad=-0.5):
    arc = FancyArrowPatch(
        (s, df["filbench_score"].iloc[s] + y_offset),
        (e, df["filbench_score"].iloc[e] + y_offset),
        arrowstyle="->", connectionstyle=f"arc3,rad={rad}",
        color="black", linewidth=1.5, mutation_scale=20)
    ax.add_patch(arc)
    mx = ((s + e) / 2.0) + text_x_offset
    my = df["filbench_score"].iloc[s:e + 1].max() + text_y_offset
    ax.text(mx, my, text, ha="center", va="bottom",
            fontdict={"size": 17}, multialignment="left")


df = pd.DataFrame({"filbench_score": [47.2, 47.7, 48.2, 49.5, 49.7, 51.4, 53.0]})

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(df))
colors = ["#f8ecda" if i <= 1 else BLUE for i in range(len(df))]
hatches = ["" for _ in range(len(df))]
bars = ax.bar(x, df["filbench_score"], color=colors, width=0.7, edgecolor="black", linewidth=1.5)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
for bar, sc in zip(bars, df["filbench_score"]):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{sc:.1f}",
            ha="center", va="bottom", fontdict={"size": FONT_SIZES.get("medium")})

ax.set_xticks(x); ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)
ax.set_ylim([40, 60]); ax.set_yticks([40, 45, 50, 55, 60])
T = ax.get_xaxis_transform()
ax.text(0, -0.02, "None", ha="center", va="top", transform=T, fontdict={"size": FONT_SIZES.get("medium")})
ax.text(1, -0.02, "GPT-4o", ha="center", va="top", transform=T, fontdict={"size": FONT_SIZES.get("medium")})
ax.text(2, -0.02, "Aya Exp", ha="center", va="top", transform=T, fontdict={"size": FONT_SIZES.get("medium")})
d = "-" * 8
ax.text(4.5, -0.02, r"$\vert$" + d + " Gemma 3 27B " + d + r"$\vert$", ha="center", va="top",
        transform=T, fontdict={"size": FONT_SIZES.get("medium")})

add_arc_annotation(ax, df, 0, 1, "+Use synthetic\npipeline", text_x_offset=-0.10, text_y_offset=4.6)
add_arc_annotation(ax, df, 1, 2, "+Better\nteacher", text_y_offset=4.6)
add_arc_annotation(ax, df, 2, 3, "+Match\nfamily")
add_arc_annotation(ax, df, 3, 4, r"+Scale data" "\n" r"(10k$\rightarrow$25k)", text_y_offset=4.6)
add_arc_annotation(ax, df, 4, 5, r"+Scale size" "\n" r"(4B$\rightarrow$12B)", text_x_offset=0.10)
add_arc_annotation(ax, df, 5, 6, r"+Scale size" "\n" r"(12B$\rightarrow$27B)", text_x_offset=0.25)

ax.set_xlabel("Teacher Model", fontsize=18, labelpad=25)
ax.set_ylabel("FilBench Score", fontsize=18)
for s in ax.spines.values():
    s.set_visible(True); s.set_color("black")

plt.tight_layout()
Path("plot_outputs").mkdir(exist_ok=True)
plt.savefig("plot_outputs/tgl_ablation_filbench_scores.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/tmp/prev_ablation.png", dpi=130, bbox_inches="tight")
print("wrote plot_outputs/tgl_ablation_filbench_scores.pdf")
