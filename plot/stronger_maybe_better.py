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

    # Prepare data
    df = prepare_data(args.pg_scores_path, args.teacher_perf_path)
    logging.info(f"Models: {df['teacher_model'].nunique()}, Languages: {df['target_lang'].nunique()}, Observations: {len(df)}")  # fmt: skip

    # Model 1: Scale only
    logging.info("Model 1: Scale only")
    result_size = model_1_scale_only(df)
    report_results(result_size, "log_model_size")

    # Model 2: Benchmark performance only
    logging.info("Model 2: Benchmark performance only")
    result_perf = model_2_benchmark_perf_only(df)
    report_results(result_perf, "benchmark_performance")

    # Model 3: Combined
    logging.info("Model 3: Combined")
    result_combined = model_3_combined(df)

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

    # Summary statistics
    df_summary = df.groupby(
        ["teacher_model", "beautiful_name", "model_size", "benchmark_performance"],
        as_index=False,
    )["pg_score"].mean()
    df_summary = df_summary.sort_values("model_size")
    df_summary.columns = ["Model ID", "Model", "Size (B)", "Benchmark (avg_all)", "Avg PG-Score"]  # fmt: skip

    print(f"\n{df_summary[['Model', 'Size (B)', 'Benchmark (avg_all)', 'Avg PG-Score']].to_string(index=False)}")  # fmt: skip

    # Correlation between size and benchmark performance
    size_perf_corr = df_summary["Size (B)"].corr(df_summary["Benchmark (avg_all)"])
    logging.info(f"Size-Performance correlation: r = {size_perf_corr:.3f}")

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
        logging.info(f"Saved results to {args.output_path}")


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
    df = df_merged[df_merged["parameter_size"] != "Unknown"].copy()
    df["model_size"] = df["parameter_size"].astype(float)
    df["log_model_size"] = np.log(df["model_size"])
    df["benchmark_performance"] = df["avg_all"]
    df = df.dropna(subset=["benchmark_performance"])

    return df


def model_1_scale_only(df_plot: pd.DataFrame):
    """Fit mixed-effects model with log(model_size) as predictor."""
    model = smf.mixedlm(
        "pg_score ~ log_model_size",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    return model.fit(method="lbfgs")


def model_2_benchmark_perf_only(df_plot: pd.DataFrame):
    """Fit mixed-effects model with benchmark_performance as predictor."""
    model = smf.mixedlm(
        "pg_score ~ benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    return model.fit(method="lbfgs")


def model_3_combined(df_plot: pd.DataFrame):
    """Fit mixed-effects model with both log(model_size) and benchmark_performance."""
    model = smf.mixedlm(
        "pg_score ~ log_model_size + benchmark_performance",
        df_plot,
        groups=df_plot["teacher_model"],
        re_formula="1",
    )
    return model.fit(method="lbfgs")


def report_results(result, predictor: str):
    """Report the results of a mixed-effects model."""
    logging.info(
        f"β = {result.params[predictor]:.4f}, SE = {result.bse[predictor]:.4f}, p = {result.pvalues[predictor]:.4f}"
    )


if __name__ == "__main__":
    main()
