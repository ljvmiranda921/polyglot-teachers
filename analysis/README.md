# Analysis & plots

Reproduce every figure in [`plot_outputs/`](../plot_outputs). Run from the **repo
root** as modules. Shared styling (Helvetica font via LaTeX, colors, output dir)
lives in [`utils/plot_theme.py`](utils/plot_theme.py); needs a LaTeX install with the
`helvet` and `sansmath` packages.

```bash
# tgl_ablation_filbench_scores.pdf
python -m analysis.ablation_results --input_dir results/tgl_ablations --figsize 9,4

# translation_ablation.pdf
python -m analysis.translation_ablation --input_path results/pg_scores_translate_abl.jsonl --figsize 8,12

# data_scale_effect.pdf  (--benchmark_only = committed single-panel; drop it for the two-panel version)
python -m analysis.data_scale_effect --benchmark_only --figsize 6,6

# language_correl.pdf
python -m analysis.language_correl --input_path data/mtep-cache/pg_scores_with_stderr.jsonl --figsize 6,6

# pgscore_robustness_heatmap.pdf
python -m analysis.pgscore_robustness -i results/pg_scores_base_olmo3.jsonl

# base_model_correlation_heatmap.pdf
python -m analysis.base_model_effect --reference_result results/pg_scores_base_olmo3.jsonl \
  -b "Llama 3 8B::results/pg_scores_base_llama3-8b.jsonl" \
  -b "Gemma 3 4B::results/pg_scores_base_gemma3-4b.jsonl" \
  -b "Qwen 3 8B::results/pg_scores_base_qwen3-8b.jsonl"

# pca_loading_factors.pdf + pca_predicted_vs_actual_linear.pdf  (--models linear pins the committed filename)
python -m analysis.principal_components --intrinsic_dir data/csd3 --benchmark_path results/pg_scores_base_olmo3.jsonl --models linear
```

No figure: `model_scale.py` (`--input_path <pg_scores>.jsonl`), `stronger_maybe_better.py`
(regression tables to stdout), `inspect_dataset.py` (dataset stats for a HF dataset ID).
