import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.principal_components import prepare_dataframe
from analysis.utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

plt.rcParams.update(PLOT_PARAMS)

FEATURE_COLS = [
    "prompts_distinct_ri",
    "responses_distinct_ri",
    "average_perplexity",
    "average_rubric_score",
    "prompts_average_length",
    "responses_average_length",
]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Leave-one-teacher-out / leave-one-language-out CV for the PCA regression on intrinsic metrics")
    parser.add_argument("--intrinsic_dir", type=Path, required=True, help="Directory containing intrinsic metrics JSON files (e.g., data/csd3/)")
    parser.add_argument("--benchmark_path", type=Path, required=True, help="JSONL file with benchmark scores. Must contain `teacher_model`, `target_lang`, and the results key.")
    parser.add_argument("--results_key", type=str, default="pg_score", help="Key in benchmark JSONL file to use as target variable.")
    parser.add_argument("--n_components", type=int, default=4, help="Number of principal components to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the 80/20 sanity-check split.")
    # fmt: on
    return parser.parse_args()


def make_model(n_components: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("linear", LinearRegression()),
        ]
    )


def leave_one_group_out(
    df: pd.DataFrame, group_col: str, results_key: str, n_components: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df with `y_pred` column, per-group metrics table).

    The scaler and PCA are fit on each training fold only, so nothing from
    the held-out group leaks into the transform.
    """
    y_all = df[results_key].values
    global_mean = y_all.mean()

    y_pred = np.full(len(df), np.nan)
    for group in sorted(df[group_col].unique()):
        held_out = (df[group_col] == group).values
        model = make_model(n_components)
        model.fit(df.loc[~held_out, FEATURE_COLS].values, y_all[~held_out])
        y_pred[held_out] = model.predict(df.loc[held_out, FEATURE_COLS].values)

    df = df.assign(y_pred=y_pred)

    rows = []
    for group, group_df in df.groupby(group_col):
        residuals = group_df[results_key] - group_df["y_pred"]
        # Per-group R^2 uses the global mean for SS_tot (same convention as
        # the pooled R^2); a per-group mean would be undefined for n=1.
        ss_tot = ((group_df[results_key] - global_mean) ** 2).sum()
        rows.append(
            {
                group_col: group,
                "n": len(group_df),
                "R^2": 1 - (residuals**2).sum() / ss_tot,
                "RMSE": np.sqrt((residuals**2).mean()),
            }
        )
    return df, pd.DataFrame(rows)


def pooled_metrics(df: pd.DataFrame, results_key: str) -> tuple[float, float]:
    y_true = df[results_key].values
    return (
        r2_score(y_true, df["y_pred"].values),
        np.sqrt(mean_squared_error(y_true, df["y_pred"].values)),
    )


def plot_loto_predicted_vs_actual(df: pd.DataFrame, results_key: str, output_path: Path):
    teachers = sorted(df["teacher_model"].unique())
    colors = [
        COLORS["cambridge_blue"],
        COLORS["crest"],
        COLORS["warm_purple"],
        COLORS["dark_indigo"],
        COLORS["judge_yellow"],
        COLORS["cherry"],
        COLORS["green"],
        COLORS["warm_blue"],
        COLORS["dark_crest"],
        COLORS["slate_3"],
    ]
    markers = ["o", "s", "^", "v", "p", "D", "<", ">", "h", "X"]

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, teacher in enumerate(teachers):
        group_df = df[df["teacher_model"] == teacher]
        ax.scatter(
            group_df[results_key],
            group_df["y_pred"],
            alpha=0.8,
            s=120,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            edgecolors=COLORS["dark_blue"],
            linewidth=1.5,
            label=teacher,
        )

    lo = min(df[results_key].min(), df["y_pred"].min())
    hi = max(df[results_key].max(), df["y_pred"].max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["slate_3"], linewidth=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Actual Score", fontsize=20)
    ax.set_ylabel("Predicted Score (LOTO)", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=2,
        fontsize=12,
        columnspacing=0.5,
        handletextpad=0.4,
    )
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    logging.info(f"Saved LOTO predicted vs actual plot to {output_path}")


def main():
    args = get_args()

    df = prepare_dataframe(args.intrinsic_dir, args.benchmark_path, args.results_key)
    # Perplexity is heavy-tailed; compress it before z-scoring. Row-wise and
    # deterministic, so applying it before the CV split leaks nothing.
    df["average_perplexity"] = -np.log1p(df["average_perplexity"])
    logging.info(
        f"Loaded {len(df)} samples: {df['teacher_model'].nunique()} teachers, "
        f"{df['target_lang'].nunique()} languages, target `{args.results_key}`"
    )

    X = df[FEATURE_COLS].values
    y = df[args.results_key].values

    pca_probe = PCA(n_components=args.n_components)
    pca_probe.fit(StandardScaler().fit_transform(X))
    logging.info(
        f"{args.n_components} PCs on the full data explain "
        f"{pca_probe.explained_variance_ratio_.sum():.1%} of variance"
    )

    # Sanity check: original-style random 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    model = make_model(args.n_components)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    split_r2 = r2_score(y_test, y_pred_test)
    split_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"\n=== Random 80/20 split (seed={args.seed}) ===")
    print(f"Train n={len(y_train)}, test n={len(y_test)}")
    print(f"R^2: {split_r2:.4f}, RMSE: {split_rmse:.4f}")

    # Leave-one-teacher-out
    df_loto, loto_table = leave_one_group_out(
        df, "teacher_model", args.results_key, args.n_components
    )
    loto_r2, loto_rmse = pooled_metrics(df_loto, args.results_key)
    print("\n=== Leave-one-teacher-out CV ===")
    print(f"Pooled R^2: {loto_r2:.4f}, pooled RMSE: {loto_rmse:.4f}")
    print("\nPer-teacher metrics (R^2 vs global mean):")
    print(loto_table.sort_values("R^2").to_string(index=False))

    # Leave-one-language-out
    df_lolo, lolo_table = leave_one_group_out(
        df, "target_lang", args.results_key, args.n_components
    )
    lolo_r2, lolo_rmse = pooled_metrics(df_lolo, args.results_key)
    print("\n=== Leave-one-language-out CV ===")
    print(f"Pooled R^2: {lolo_r2:.4f}, pooled RMSE: {lolo_rmse:.4f}")
    print("\nPer-language metrics (R^2 vs global mean):")
    print(lolo_table.sort_values("R^2").to_string(index=False))

    print("\n=== Comparison ===")
    comparison = pd.DataFrame(
        [
            {"Evaluation": "Random 80/20 split", "R^2": split_r2, "RMSE": split_rmse},
            {"Evaluation": "Leave-one-teacher-out", "R^2": loto_r2, "RMSE": loto_rmse},
            {"Evaluation": "Leave-one-language-out", "R^2": lolo_r2, "RMSE": lolo_rmse},
        ]
    )
    print(comparison.to_string(index=False))

    plot_loto_predicted_vs_actual(
        df_loto, args.results_key, OUTPUT_DIR / "pca_loto_predicted_vs_actual.pdf"
    )


if __name__ == "__main__":
    main()
