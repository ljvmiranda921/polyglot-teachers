import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from plot.utils.metadata import MODEL_INFORMATION

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Analyze whether model scale and teacher performance predict PG-score")
    parser.add_argument("--pg_scores_path", type=Path, required=True, help="PG scores JSONL file. Must contain `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--teacher_perf_path", type=Path, required=True, help="Teacher performance JSONL file. Must contain `model_name` and performance metrics.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to save the results (CSV). If not provided, results are printed to stdout.")
    # fmt: on
    return parser.parse_args()


def prepare_data(pg_scores_path: Path, teacher_perf_path: Path) -> pd.DataFrame:
    """Prepare and merge data from PG scores and teacher performance files."""
    df_pg = pd.read_json(pg_scores_path, lines=True)

    df_teacher = pd.read_json(teacher_perf_path, lines=True)
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

    return df_plot


def model_1_scale_only(df_plot: pd.DataFrame):
    """Fit mixed-effects model with log(model_size) as predictor."""
    logger.info("=" * 80)
    logger.info("1. MODEL SCALE → PG-SCORE")
    logger.info("=" * 80)
    model = smf.mixedlm(
        "pg_score ~ log_model_size",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result = model.fit(method="lbfgs")
    return result


def model_2_benchmark_perf_only(df_plot: pd.DataFrame):
    """Fit mixed-effects model with benchmark_performance as predictor."""
    logger.info("=" * 80)
    logger.info("2. BENCHMARK PERFORMANCE (avg_all) → PG-SCORE")
    logger.info("=" * 80)
    model = smf.mixedlm(
        "pg_score ~ benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result = model.fit(method="lbfgs")
    return result


def model_3_combined(df_plot: pd.DataFrame):
    """Fit mixed-effects model with both log(model_size) and benchmark_performance."""
    logger.info("=" * 80)
    logger.info("3. COMBINED MODEL")
    logger.info("=" * 80)
    model = smf.mixedlm(
        "pg_score ~ log_model_size + benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    result = model.fit(method="lbfgs")
    return result


def report_results(result, predictor: str):
    """Report the results of a mixed-effects model."""
    logger.info(f"β = {result.params[predictor]:.4f}, SE = {result.bse[predictor]:.4f}, p = {result.pvalues[predictor]:.4f}")


def main():
    args = get_args()

    # Prepare data
    df_plot = prepare_data(args.pg_scores_path, args.teacher_perf_path)

    logger.info("=" * 80)
    logger.info("ANALYSIS: Does Stronger Mean Better?")
    logger.info("=" * 80)
    logger.info(f"PG scores file: {args.pg_scores_path}")
    logger.info(f"Teacher performance file: {args.teacher_perf_path}")
    logger.info(f"Total models analyzed: {df_plot['teacher_model'].nunique()}")
    logger.info(f"Total languages: {df_plot['target_lang'].nunique()}")
    logger.info(f"Total observations: {len(df_plot)}")

    # =========================================================================
    # Mixed-Effects Models
    # =========================================================================

    # Model 1: Scale only
    result_size = model_1_scale_only(df_plot)
    report_results(result_size, "log_model_size")

    # Model 2: Benchmark performance only
    result_perf = model_2_benchmark_perf_only(df_plot)
    report_results(result_perf, "benchmark_performance")

    # Model 3: Combined
    result_combined = model_3_combined(df_plot)

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
    logger.info("=" * 80)
    logger.info("4. MODEL SUMMARY")
    logger.info("=" * 80)

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

    print(f"\n{df_summary[['Model', 'Size (B)', 'Benchmark (avg_all)', 'Avg PG-Score']].to_string(index=False)}")

    # Correlation between size and benchmark performance
    size_perf_corr = df_summary["Size (B)"].corr(df_summary["Benchmark (avg_all)"])
    logger.info(f"Correlation between model size and benchmark performance: r = {size_perf_corr:.3f}")
    logger.info("=" * 80)

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
        logger.info(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
