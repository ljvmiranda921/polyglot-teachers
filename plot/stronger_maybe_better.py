import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from plot.utils.metadata import MODEL_INFORMATION


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Analyze whether model scale and teacher performance predict PG-score")
    parser.add_argument("--pg_scores_path", type=Path, required=True, help="PG scores JSONL file. Must contain `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--teacher_perf_path", type=Path, required=True, help="Teacher performance JSONL file. Must contain `model_name` and performance metrics.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to save the results (CSV). If not provided, results are printed to stdout.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df_pg = pd.read_json(args.pg_scores_path, lines=True)

    df_teacher = pd.read_json(args.teacher_perf_path, lines=True)
    df_teacher["teacher_model"] = df_teacher["model_name"].str.split("/").str[-1]

    df_models = pd.DataFrame([m.model_dump() for m in MODEL_INFORMATION])
    df_models["teacher_model"] = df_models["name"].str.split("/").str[-1]

    df_merged = df_pg.merge(
        df_models[["teacher_model", "parameter_size", "beautiful_name"]],
        on="teacher_model",
        how="left",
    ).merge(
        df_teacher[["teacher_model", "avg_all"]],
        on="teacher_model",
        how="left",
    )

    # Filter out unknown parameter sizes
    df_plot = df_merged[df_merged["parameter_size"] != "Unknown"].copy()
    df_plot["model_size"] = df_plot["parameter_size"].astype(float)
    df_plot["log_model_size"] = np.log(df_plot["model_size"])
    df_plot["benchmark_performance"] = df_plot["avg_all"]
    df_plot = df_plot.dropna(subset=["benchmark_performance"])

    print("\n" + "=" * 80)
    print("ANALYSIS: Does Stronger Mean Better?")
    print("=" * 80)
    print(f"\nPG scores file: {args.pg_scores_path}")
    print(f"Teacher performance file: {args.teacher_perf_path}")
    print(f"\nTotal models analyzed: {df_plot['teacher_model'].nunique()}")
    print(f"Total languages: {df_plot['target_lang'].nunique()}")
    print(f"Total observations: {len(df_plot)}")

    # =========================================================================
    # Mixed-Effects Models
    # =========================================================================

    # Model 1: Scale only
    print("\n" + "=" * 80)
    print("1. MODEL SCALE → PG-SCORE")
    print("=" * 80)
    model_size = smf.mixedlm(
        "pg_score ~ log_model_size",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result_size = model_size.fit(method="lbfgs")
    print(
        f"\nβ = {result_size.params['log_model_size']:.4f}, SE = {result_size.bse['log_model_size']:.4f}, p = {result_size.pvalues['log_model_size']:.4f}"
    )

    # Model 2: Benchmark performance only
    print("\n" + "=" * 80)
    print("2. BENCHMARK PERFORMANCE (avg_all) → PG-SCORE")
    print("=" * 80)
    model_perf = smf.mixedlm(
        "pg_score ~ benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result_perf = model_perf.fit(method="lbfgs")
    print(
        f"\nβ = {result_perf.params['benchmark_performance']:.4f}, SE = {result_perf.bse['benchmark_performance']:.4f}, p = {result_perf.pvalues['benchmark_performance']:.4f}"
    )

    # Model 3: Combined
    print("\n" + "=" * 80)
    print("3. COMBINED MODEL")
    print("=" * 80)
    model_combined = smf.mixedlm(
        "pg_score ~ log_model_size + benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result_combined = model_combined.fit(method="lbfgs")

    # Create results table as DataFrame
    results_table = pd.DataFrame(
        [
            {
                "Predictor": "log(model_size)",
                "β": f"{result_combined.params['log_model_size']:.3f}",
                "SE": f"{result_combined.bse['log_model_size']:.3f}",
                "p": f"{result_combined.pvalues['log_model_size']:.3f}",
            },
            {
                "Predictor": "benchmark_performance",
                "β": f"{result_combined.params['benchmark_performance']:.3f}",
                "SE": f"{result_combined.bse['benchmark_performance']:.3f}",
                "p": f"{result_combined.pvalues['benchmark_performance']:.3f}",
            },
        ]
    )

    # Print markdown table
    print("\n" + results_table.to_markdown(index=False))

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. MODEL SUMMARY")
    print("=" * 80)

    df_summary = df_plot.groupby(
        ["teacher_model", "beautiful_name", "model_size", "benchmark_performance"],
        as_index=False,
    )["pg_score"].mean()
    df_summary = df_summary.sort_values("model_size")
    df_summary.columns = [
        "Model ID",
        "Model",
        "Size (B)",
        "Benchmark (avg_all)",
        "Avg PG-Score",
    ]

    print(
        f"\n{df_summary[['Model', 'Size (B)', 'Benchmark (avg_all)', 'Avg PG-Score']].to_string(index=False)}"
    )

    # Correlation between size and benchmark performance
    size_perf_corr = df_summary["Size (B)"].corr(df_summary["Benchmark (avg_all)"])
    print(
        f"\nCorrelation between model size and benchmark performance: r = {size_perf_corr:.3f}"
    )

    print("=" * 80 + "\n")

    # =========================================================================
    # Save results if requested
    # =========================================================================
    if args.output_path:
        results_data = []

        results_data.append(
            {
                "model": "scale_only",
                "predictor": "log_model_size",
                "beta": result_size.params["log_model_size"],
                "se": result_size.bse["log_model_size"],
                "t": result_size.tvalues["log_model_size"],
                "p": result_size.pvalues["log_model_size"],
                "aic": result_size.aic,
                "bic": result_size.bic,
            }
        )

        results_data.append(
            {
                "model": "performance_only",
                "predictor": "benchmark_performance",
                "beta": result_perf.params["benchmark_performance"],
                "se": result_perf.bse["benchmark_performance"],
                "t": result_perf.tvalues["benchmark_performance"],
                "p": result_perf.pvalues["benchmark_performance"],
                "aic": result_perf.aic,
                "bic": result_perf.bic,
            }
        )

        results_data.append(
            {
                "model": "combined",
                "predictor": "log_model_size",
                "beta": result_combined.params["log_model_size"],
                "se": result_combined.bse["log_model_size"],
                "t": result_combined.tvalues["log_model_size"],
                "p": result_combined.pvalues["log_model_size"],
                "aic": result_combined.aic,
                "bic": result_combined.bic,
            }
        )
        results_data.append(
            {
                "model": "combined",
                "predictor": "benchmark_performance",
                "beta": result_combined.params["benchmark_performance"],
                "se": result_combined.bse["benchmark_performance"],
                "t": result_combined.tvalues["benchmark_performance"],
                "p": result_combined.pvalues["benchmark_performance"],
                "aic": result_combined.aic,
                "bic": result_combined.bic,
            }
        )

        df_results = pd.DataFrame(results_data)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(args.output_path, index=False)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
