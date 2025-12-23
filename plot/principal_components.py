import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

from utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

plt.rcParams.update(PLOT_PARAMS)

FEATURE_NAMES = {
    "prompts_distinct_ri": "Distinct\nPrompts",
    "responses_distinct_ri": "Distinct\nResponses",
    "average_perplexity": "Perplexity",
    "average_rubric_score": "Rubric Score\n(M-Prometheus)",
    "prompts_average_length": "Avg. Prompt\nLength",
    "responses_average_length": "Avg. Response\nLength",
}

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "cs": "Czech",
    "es": "Spanish",
    "id": "Indonesian",
    "ja": "Japanese",
}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="PCA analysis on intrinsic metrics to predict benchmark performance")
    parser.add_argument("--intrinsic_dir", type=Path, required=True, help="Directory containing intrinsic metrics JSON files (e.g., data/csd3/)")
    parser.add_argument("--benchmark_path", type=Path, required=True, help="JSONL file with benchmark scores. Must contain `teacher_model`, `target_lang`, and `result`.")
    parser.add_argument("--n_components", type=int, default=None, help="Number of principal components to use. If not specified, uses all components.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to save results (CSV). If not provided, results are printed to stdout.")
    parser.add_argument("--results_key", type=str, default="result", help="Key in benchmark JSONL file to use as target variable.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = prepare_dataframe(args.intrinsic_dir, args.benchmark_path, args.results_key)
    logging.info(
        f"Loaded {len(df)} samples with both intrinsic metrics and benchmark scores"
    )

    feature_cols = [
        "prompts_distinct_ri",
        "responses_distinct_ri",
        "average_perplexity",
        "average_rubric_score",
        "prompts_average_length",
        "responses_average_length",
    ]
    X = df[feature_cols].values
    y = df[args.results_key].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = args.n_components or X.shape[1]
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    logging.info(f"PCA with {n_components} components")
    logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logging.info(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")  # fmt: skip

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_pca, y)
        y_pred = model.predict(X_pca)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results[name] = {
            "model": model,
            "r2": r2,
            "rmse": rmse,
            "y_pred": y_pred,
        }
        logging.info(f"{name} - R^2: {r2:.4f}, RMSE: {rmse:.4f}")

    best_model_name = max(results, key=lambda k: results[k]["r2"])
    model = results[best_model_name]["model"]
    r2 = results[best_model_name]["r2"]
    y_pred = results[best_model_name]["y_pred"]
    logging.info(f"Best model: {best_model_name}")

    print("\n=== PCA Components ===")
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_cols,
    )
    print(components_df.to_string())

    print(f"\n=== Model Comparison ===")
    comparison_df = pd.DataFrame(
        [
            {"Model": name, "R^2": res["r2"], "RMSE": res["rmse"]}
            for name, res in results.items()
        ]
    ).sort_values("R^2", ascending=False)
    print(comparison_df.to_string(index=False))

    for model_name, res in results.items():
        print(f"\n=== {model_name} Regression ===")
        print(f"R^2 score: {res['r2']:.4f}")
        print(f"RMSE: {res['rmse']:.4f}")
        print(f"Intercept: {res['model'].intercept_:.4f}")
        print("\nCoefficients:")
        for i, coef in enumerate(res["model"].coef_):
            print(f"  PC{i+1}: {coef:.4f}")

    print(f"\n=== Explained Variance ===")
    variance_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Variance": pca.explained_variance_ratio_,
            "Cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    print(variance_df.to_string(index=False))

    # Plot loading factors heatmap
    heatmap_path = OUTPUT_DIR / "pca_loading_factors.pdf"
    plot_loading_factors_heatmap(pca, feature_cols, n_components, heatmap_path)
    # Plot predicted vs actual for best model
    pred_vs_actual_path = OUTPUT_DIR / f"pca_predicted_vs_actual_{best_model_name.lower()}.pdf"  # fmt: skip
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plot_predicted_vs_actual(y, y_pred, r2, rmse, best_model_name, pred_vs_actual_path, df["target_lang"])  # fmt: skip

    if args.output_path:
        results = {
            "n_components": n_components,
            "r2_score": r2,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
            "feature_names": feature_cols,
            "pca_components": pca.components_.tolist(),
        }

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved results to {args.output_path}")


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    r2,
    rmse,
    model_name,
    output_path,
    languages=None,
    norm_range=(0.3, 0.5),
):
    # Apply custom range normalization
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    range_min, range_max = norm_range

    # Normalize to custom range: new_val = (val - min) / (max - min) * (range_max - range_min) + range_min
    y_true_norm = (y_true - y_min) / (y_max - y_min) * (
        range_max - range_min
    ) + range_min
    y_pred_norm = (y_pred - y_min) / (y_max - y_min) * (
        range_max - range_min
    ) + range_min

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by language if provided
    unique_langs = sorted(languages.unique())
    colors = [
        COLORS["warm_blue"],
        COLORS["warm_green"],
        COLORS["warm_cherry"],
        COLORS["warm_purple"],
        COLORS["warm_indigo"],
        COLORS["warm_crest"],
    ]
    lang_colors = {lang: colors[i % len(colors)] for i, lang in enumerate(unique_langs)}

    for lang in unique_langs:
        mask = languages == lang
        lang_label = LANGUAGE_NAMES.get(lang, lang)
        ax.scatter(
            y_true_norm[mask],
            y_pred_norm[mask],
            alpha=0.6,
            s=100,
            color=lang_colors[lang],
            edgecolors=COLORS["dark_blue"],
            linewidth=1.5,
            label=lang_label,
        )

    # Perfect prediction line (y=x)
    range_min, range_max = norm_range
    ax.plot(
        [range_min, range_max],
        [range_min, range_max],
        "--",
        color=COLORS["slate_3"],
        linewidth=2,
    )

    ax.set_xlabel("Actual Benchmark Score")
    ax.set_ylabel("Predicted Benchmark Score")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=3,
    )

    # Report R^2 and RMSE
    textstr = f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.3f}"
    props = dict(
        boxstyle="round",
        facecolor=COLORS["white"],
        edgecolor=COLORS["slate_3"],
        alpha=0.9,
        linewidth=1.5,
    )
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    logging.info(f"Saved predicted vs actual plot to {output_path}")


def plot_loading_factors_heatmap(pca, feature_names, n_components, output_path):
    # Map feature names to beautiful labels
    beautiful_names = [FEATURE_NAMES[name] for name in feature_names]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=beautiful_names,
    )

    # Create custom colormap using Cambridge colors
    cmap = LinearSegmentedColormap.from_list(
        "cambridge_diverging",
        [COLORS["cherry"], COLORS["white"], COLORS["green"]],
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    heatmap = sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        cbar_kws={
            "label": "Loading Strength",
            "orientation": "horizontal",
            "shrink": 0.8,
            "pad": 0.1,
        },
        ax=ax,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", va="center")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    logging.info(f"Saved loading factors heatmap to {output_path}")


def prepare_dataframe(
    intrinsic_dir: Path, benchmark_path: Path, results_key: str
) -> pd.DataFrame:
    df_intrinsic = load_intrinsic_metrics(intrinsic_dir)
    df_benchmark = pd.read_json(benchmark_path, lines=True)

    df = df_intrinsic.merge(
        df_benchmark[["teacher_model", "target_lang", results_key]],
        on=["teacher_model", "target_lang"],
        how="inner",
    )
    return df


def load_intrinsic_metrics(intrinsic_dir: Path) -> pd.DataFrame:
    records = []
    json_files = list(intrinsic_dir.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {intrinsic_dir}")

    for json_file in json_files:
        filename = json_file.stem
        parts = filename.split("_")

        if len(parts) < 3:
            logging.warning(f"Skipping {json_file.name}: unexpected filename format")
            continue

        dataset_lang = parts[0].split("-")[-1]
        teacher_model_full = "_".join(parts[1:-2])

        if "__" in teacher_model_full:
            teacher_model = teacher_model_full.split("__", 1)[1]
        else:
            teacher_model = teacher_model_full

        with open(json_file) as f:
            data = json.load(f)

        record = {
            # fmt: off
            "teacher_model": teacher_model,
            "target_lang": dataset_lang,
            "prompts_distinct_ri": data.get("distinct_ri", {}).get("prompts_distinct_ri"),
            "responses_distinct_ri": data.get("distinct_ri", {}).get("responses_distinct_ri"),
            "average_perplexity": data.get("perplexity", {}).get("average_perplexity"),
            "average_rubric_score": data.get("reward_model", {}).get("average_rubric_score"),
            "prompts_average_length": data.get("length", {}).get("prompts_average_length"),
            "responses_average_length": data.get("length", {}).get("responses_average_length"),
            # fmt: on
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.dropna()
    logging.info(f"Loaded {len(df)} complete records with all intrinsic metrics")

    return df


if __name__ == "__main__":
    main()
