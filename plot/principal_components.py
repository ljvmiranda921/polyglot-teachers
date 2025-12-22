import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="PCA analysis on intrinsic metrics to predict benchmark performance")
    parser.add_argument("--intrinsic_dir", type=Path, required=True, help="Directory containing intrinsic metrics JSON files (e.g., data/csd3/)")
    parser.add_argument("--benchmark_path", type=Path, required=True, help="JSONL file with benchmark scores. Must contain `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--n_components", type=int, default=None, help="Number of principal components to use. If not specified, uses all components.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to save results (CSV). If not provided, results are printed to stdout.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = prepare_dataframe(args.intrinsic_dir, args.benchmark_path)
    logging.info(
        f"Loaded {len(df)} samples with both intrinsic metrics and benchmark scores"
    )
    breakpoint()

    feature_cols = [
        "prompts_distinct_ri",
        "responses_distinct_ri",
        "average_perplexity",
        "average_rubric_score",
        "prompts_average_length",
        "responses_average_length",
    ]
    X = df[feature_cols].values
    y = df["pg_score"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = args.n_components or X.shape[1]
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    logging.info(f"PCA with {n_components} components")
    logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logging.info(
        f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}"
    )

    model = LinearRegression()
    model.fit(X_pca, y)
    y_pred = model.predict(X_pca)
    r2 = r2_score(y, y_pred)

    logging.info(f"R² score: {r2:.4f}")
    logging.info(f"Model coefficients: {model.coef_}")
    logging.info(f"Model intercept: {model.intercept_:.4f}")

    print("\n=== PCA Components ===")
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_cols,
    )
    print(components_df.to_string())

    print(f"\n=== Regression Results ===")
    print(f"R^2 score: {r2:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print("\nCoefficients:")
    for i, coef in enumerate(model.coef_):
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


def prepare_dataframe(intrinsic_dir: Path, benchmark_path: Path) -> pd.DataFrame:
    df_intrinsic = load_intrinsic_metrics(intrinsic_dir)
    df_benchmark = pd.read_json(benchmark_path, lines=True)

    df = df_intrinsic.merge(
        df_benchmark[["teacher_model", "target_lang", "pg_score"]],
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
        teacher_model = "_".join(parts[1:-2])

        with open(json_file) as f:
            data = json.load(f)

        record = {
            "teacher_model": teacher_model,
            "target_lang": dataset_lang,
        }

        if "distinct_ri" in data:
            record["prompts_distinct_ri"] = data["distinct_ri"].get(
                "prompts_distinct_ri"
            )
            record["responses_distinct_ri"] = data["distinct_ri"].get(
                "responses_distinct_ri"
            )

        if "perplexity" in data:
            record["average_perplexity"] = data["perplexity"].get("average_perplexity")

        if "reward_model" in data:
            record["average_rubric_score"] = data["reward_model"].get(
                "average_rubric_score"
            )

        if "length" in data:
            record["prompts_average_length"] = data["length"].get(
                "prompts_average_length"
            )
            record["responses_average_length"] = data["length"].get(
                "responses_average_length"
            )

        records.append(record)

    df = pd.DataFrame(records)
    df = df.dropna()
    logging.info(f"Loaded {len(df)} complete records with all intrinsic metrics")

    return df


if __name__ == "__main__":
    main()
