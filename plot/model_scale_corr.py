import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from plot.utils.metadata import MODEL_INFORMATION


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Compute correlation between model scale and PG-score")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to analyze. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path to save the correlation results (CSV). If not provided, results are printed to stdout.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Load data
    df = pd.read_json(args.input_path, lines=True)

    # Merge with model information to get parameter sizes
    df_models = pd.DataFrame([m.model_dump() for m in MODEL_INFORMATION])
    df_models["teacher_model"] = df_models["name"].str.split("/").str[-1]
    df_plot = df.merge(
        df_models[["teacher_model", "parameter_size", "beautiful_name"]],
        on="teacher_model",
        how="left",
    )

    # Filter out unknown parameter sizes
    df_plot = df_plot[df_plot["parameter_size"] != "Unknown"].copy()
    df_plot["model_size"] = df_plot["parameter_size"].astype(float)

    # Compute overall correlation (all data points)
    pearson_r, pearson_p = stats.pearsonr(df_plot["model_size"], df_plot["pg_score"])
    spearman_r, spearman_p = stats.spearmanr(df_plot["model_size"], df_plot["pg_score"])

    # Compute correlation using model averages
    df_avg = df_plot.groupby(["teacher_model", "model_size", "beautiful_name"], as_index=False)["pg_score"].mean()  # fmt: skip
    pearson_r_avg, pearson_p_avg = stats.pearsonr(df_avg["model_size"], df_avg["pg_score"])  # fmt: skip
    spearman_r_avg, spearman_p_avg = stats.spearmanr(df_avg["model_size"], df_avg["pg_score"])  # fmt: skip

    # Compute mixed-effects model (if statsmodels available)
    mixed_model_result = None
    if HAS_STATSMODELS:
        # Use log(model_size) for better linearity
        df_plot["log_model_size"] = np.log(df_plot["model_size"])

        # Mixed model with random intercepts for both model and language
        # Formula: pg_score ~ log_model_size + (1|teacher_model) + (1|target_lang)
        try:
            model = smf.mixedlm(
                "pg_score ~ log_model_size",
                df_plot,
                groups=df_plot["teacher_model"],
                re_formula="1"
            )
            mixed_result = model.fit(method="lbfgs")
            mixed_model_result = {
                "coef": mixed_result.params["log_model_size"],
                "se": mixed_result.bse["log_model_size"],
                "pvalue": mixed_result.pvalues["log_model_size"],
                "tvalue": mixed_result.tvalues["log_model_size"],
                "converged": mixed_result.converged,
            }
        except Exception as e:
            print(f"\nWarning: Mixed model fitting failed: {e}")
            mixed_model_result = None

    # Create results dataframe
    results = pd.DataFrame([
        {
            "analysis_type": "all_points",
            "n_points": len(df_plot),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        },
        {
            "analysis_type": "model_averages",
            "n_points": len(df_avg),
            "pearson_r": pearson_r_avg,
            "pearson_p": pearson_p_avg,
            "spearman_r": spearman_r_avg,
            "spearman_p": spearman_p_avg,
        },
    ])

    # Print results
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Model Scale vs PG-Score")
    print("="*80)
    print(f"\nInput file: {args.input_path}")
    print(f"\nTotal models analyzed: {df_plot['teacher_model'].nunique()}")
    print(f"Total languages: {df_plot['target_lang'].nunique()}")

    print("\n" + "-"*80)
    print("1. ALL DATA POINTS (per model-language pair)")
    print("-"*80)
    print(f"   N = {len(df_plot)}")
    print(f"   Pearson's r  = {pearson_r:7.4f} (p = {pearson_p:.4e})")
    print(f"   Spearman's rho = {spearman_r:7.4f} (p = {spearman_p:.4e})")

    print("\n" + "-"*80)
    print("2. MODEL AVERAGES (averaged across languages)")
    print("-"*80)
    print(f"   N = {len(df_avg)}")
    print(f"   Pearson's r  = {pearson_r_avg:7.4f} (p = {pearson_p_avg:.4e})")
    print(f"   Spearman's rho = {spearman_r_avg:7.4f} (p = {spearman_p_avg:.4e})")

    # Print mixed model results if available
    if mixed_model_result and mixed_model_result["converged"]:
        print("\n" + "-"*80)
        print("3. MIXED-EFFECTS MODEL (random intercepts by model)")
        print("-"*80)
        print(f"   Formula: pg_score ~ log(model_size) + (1|teacher_model)")
        print(f"   Coefficient = {mixed_model_result['coef']:7.4f} (SE = {mixed_model_result['se']:.4f})")
        print(f"   t-value     = {mixed_model_result['tvalue']:7.4f} (p = {mixed_model_result['pvalue']:.4e})")
        print(f"   ")
        print(f"   Interpretation: A 1-unit increase in log(model size)")
        print(f"   is associated with a {mixed_model_result['coef']:.4f} change in PG-score")
    elif not HAS_STATSMODELS:
        print("\n" + "-"*80)
        print("3. MIXED-EFFECTS MODEL")
        print("-"*80)
        print("   Not available (install statsmodels: pip install statsmodels)")

    print("\n" + "-"*80)
    print("Model Details:")
    print("-"*80)
    model_summary = df_avg.sort_values("model_size")[["beautiful_name", "model_size", "pg_score"]]
    model_summary.columns = ["Model", "Size (B)", "Avg PG-Score"]
    print(model_summary.to_string(index=False))
    print("="*80 + "\n")

    # Save to file if output path is provided
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_path, index=False)
        print(f"\nSaved correlation results to {args.output_path}")

        # Also save the model averages
        model_avg_path = args.output_path.parent / f"{args.output_path.stem}_model_averages.csv"
        df_avg.to_csv(model_avg_path, index=False)
        print(f"Saved model averages to {model_avg_path}")


if __name__ == "__main__":
    main()
