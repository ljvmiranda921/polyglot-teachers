import pandas as pd
import argparse
from pathlib import Path
import logging
import sys
from scipy.stats import spearmanr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    parser = argparse.ArgumentParser(description="Compute base model effect")
    # fmt: off
    parser.add_argument("--reference_result", type=Path, help="Path to the OLMo 3 7B results.")
    parser.add_argument("-b", "--base_model_result", action="append", type=str, help="Base model result in format <base_model>::<path/to/results.jsonl>")
    parser.add_argument("-o", "--output_path", type=Path, default="results/base_model_effect.csv", help="Path to save the results in CSV format.")
    parser.add_argument("-l", "--languages", nargs="+", type=str, default=["ar", "id", "de"], help="Language code to include in computation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    base_model_results: list[tuple[str, Path]] = [parse_base_model_input(str_input) for str_input in args.base_model_result]  # fmt: skip
    languages = args.languages
    logging.info(f"Using languages: {languages}")

    base_model_results.append(("OLMo 3 7B", args.reference_result))

    results = []
    for base_model, fp in base_model_results:
        df = pd.read_json(fp, lines=True)
        df = df[df["target_lang"].isin(languages)].reset_index(drop=True)
        base_df = df.groupby("teacher_model").agg({"pg_score": "mean", "pgr": "mean"})
        base_df["base_model"] = base_model
        base_df = base_df.sort_values(by="pg_score", ascending=False)
        print(f"========== Results for {base_model} ==========")
        print(base_df.to_markdown())
        results.append(base_df.reset_index())

    results_df = pd.concat(results).reset_index(drop=True)

    # Compute correlations
    corr_matrix = compute_correlation_matrix(results_df)
    corr_olmo3_7b = compute_correlation_on_olmo3_7b(results_df)

    # Analyze which teacher model is the clear winner
    analyze_teacher_rankings(results_df)
    breakpoint()


def parse_base_model_input(s: str) -> tuple[str, Path]:
    """Parse a string input <base_model>::<path/to/results.jsonl>"""
    base_model, path = s.split("::")
    if not Path(path).exists():
        raise ValueError(f"Cannot find file or input: {path}")
    return base_model, Path(path)


def compute_correlation_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman correlation matrix between all base models"""
    pivot_df = results_df.pivot(
        index="teacher_model", columns="base_model", values="pg_score"
    )

    # Compute pairwise Spearman correlations
    base_models = pivot_df.columns.tolist()
    n_models = len(base_models)
    corr_matrix = pd.DataFrame(index=base_models, columns=base_models, dtype=float)

    for i, model_a in enumerate(base_models):
        for j, model_b in enumerate(base_models):
            if i == j:
                corr_matrix.loc[model_a, model_b] = 1.0
            elif i < j:
                # Only compute upper triangle, then mirror
                valid_mask = pivot_df[[model_a, model_b]].notna().all(axis=1)
                valid_data = pivot_df.loc[valid_mask, [model_a, model_b]]

                if len(valid_data) > 1:
                    rho, _ = spearmanr(valid_data[model_a], valid_data[model_b])
                    corr_matrix.loc[model_a, model_b] = rho
                    corr_matrix.loc[model_b, model_a] = rho
                else:
                    corr_matrix.loc[model_a, model_b] = float("nan")
                    corr_matrix.loc[model_b, model_a] = float("nan")

    logging.info(f"Computed {n_models}x{n_models} correlation matrix")
    print("\n========== Base Model Correlation Matrix (Spearman) ==========")
    print(corr_matrix.to_markdown())

    return corr_matrix


def analyze_teacher_rankings(results_df: pd.DataFrame) -> None:
    """Analyze teacher model rankings across all base models to find a clear winner"""
    print("\n========== Teacher Model Rankings Analysis ==========")

    # Get overall statistics per teacher
    teacher_stats = results_df.groupby("teacher_model").agg({
        "pg_score": ["mean", "std", "min", "max"],
        "pgr": ["mean", "std"]
    }).round(4)

    teacher_stats.columns = ["_".join(col).strip() for col in teacher_stats.columns.values]
    teacher_stats = teacher_stats.sort_values("pg_score_mean", ascending=False)

    print("\nOverall Teacher Performance (averaged across all base models):")
    print(teacher_stats.to_markdown())

    # Count how many times each teacher ranks in top 3 across base models
    top_rankings = []
    base_models = results_df["base_model"].unique()

    for teacher in results_df["teacher_model"].unique():
        top_1_count = 0
        top_3_count = 0

        for base_model in base_models:
            base_subset = results_df[results_df["base_model"] == base_model].sort_values("pg_score", ascending=False)
            rank = base_subset[base_subset["teacher_model"] == teacher].index[0]
            rank_position = list(base_subset.index).index(rank) + 1

            if rank_position == 1:
                top_1_count += 1
            if rank_position <= 3:
                top_3_count += 1

        top_rankings.append({
            "teacher_model": teacher,
            "times_ranked_1st": top_1_count,
            "times_in_top_3": top_3_count,
            "total_base_models": len(base_models)
        })

    top_rankings_df = pd.DataFrame(top_rankings).sort_values("times_ranked_1st", ascending=False)
    print("\nRanking Frequency:")
    print(top_rankings_df.to_markdown(index=False))

    # Find the winner
    best_teacher = teacher_stats.index[0]
    best_score = teacher_stats.iloc[0]["pg_score_mean"]
    best_std = teacher_stats.iloc[0]["pg_score_std"]

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {best_teacher}")
    print(f"  - Mean PG Score: {best_score:.4f} (±{best_std:.4f})")
    print(f"  - Ranked 1st: {top_rankings_df[top_rankings_df['teacher_model']==best_teacher]['times_ranked_1st'].values[0]}/{len(base_models)} times")
    print(f"  - In Top 3: {top_rankings_df[top_rankings_df['teacher_model']==best_teacher]['times_in_top_3'].values[0]}/{len(base_models)} times")
    print(f"{'='*60}")


def compute_correlation_on_olmo3_7b(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation for each base model against OLMo 3 7B"""
    olmo_df = results_df[results_df["base_model"] == "OLMo 3 7B"].copy()
    other_base_models = results_df[results_df["base_model"] != "OLMo 3 7B"]["base_model"].unique()  # fmt: skip

    correlations = []
    for base_model in other_base_models:
        base_df = results_df[results_df["base_model"] == base_model].copy()
        merged = olmo_df.merge(
            base_df,
            on="teacher_model",
            suffixes=("_olmo", f"_{base_model.replace(' ', '_')}"),
        )
        rho, pvalue = spearmanr(merged["pg_score_olmo"], merged[f"pg_score_{base_model.replace(' ', '_')}"])  # fmt: skip

        correlations.append(
            {
                "base_model": base_model,
                "spearman_rho": rho,
                "p_value": pvalue,
                "n_teachers": len(merged),
            }
        )

        logging.info(f"Spearman rho for {base_model} vs OLMo 3 7B: {rho:.4f} (p={pvalue:.4f}, n={len(merged)})")  # fmt: skip

    correlations_df = pd.DataFrame(correlations)
    print("\n========== Spearman Correlations vs OLMo 3 7B ==========")
    print(correlations_df.to_markdown(index=False))
    return correlations_df


if __name__ == "__main__":
    main()
