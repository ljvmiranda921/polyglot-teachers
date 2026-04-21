# Experiments

To run experiments locally, simply call the bash script directly from the `jobs/` directory:

```bash
bash jobs/<job_script> [args...]
```

We also provide submission scripts for certain clusters such as [Cambridge Service for Data Driven Discovery (CSD3) cluster](https://docs.hpc.cam.ac.uk/hpc/index.html) and the [Language Technology Laboratory](https://ltl.mmll.cam.ac.uk/) Cluster.
These wrappers handle cluster-specific setup like module loading and cache configuration:

```bash
# Wilkes-CSD3
sbatch [--array=...] slurm_submit.wilkes-csd3 jobs/<job_script> [args...]

# LTL Cluster
sbatch [--array=...] slurm_submit.ltl jobs/<job_script> [args...]
```

## Examples

```bash
# Create synthetic datasets (array job for 6 languages)
sbatch --array=0-5 slurm_submit.wilkes-csd3 jobs/create_synthetic_datasets.sh

# Compute metrics (array job for 11 models × 6 languages)
sbatch --array=0-65%8 slurm_submit.wilkes-csd3 jobs/get_intrinsic_metrics.sh

# Finetune student model
sbatch slurm_submit.ltl jobs/finetune_student_model_unsloth.sh "ljvmiranda921/msde-S1-ar"
```

## Overview of Experiments

| Job script | Description |
| --- | --- |
| [create_synthetic_datasets.sh](jobs/create_synthetic_datasets.sh) | Generate synthetic data with a teacher model for each of the 6 target languages (array over languages). |
| [translate_nllb.sh](jobs/translate_nllb.sh) | Build translation-based synthetic datasets using NLLB plus a teacher model, across 3 strategies × 3 languages. |
| [get_intrinsic_metrics.sh](jobs/get_intrinsic_metrics.sh) | Compute intrinsic metrics (distinct-RI, reward model, perplexity, length) per (teacher, language) pair. |
| [finetune_student_model_unsloth.sh](jobs/finetune_student_model_unsloth.sh) | Finetune a student model with Unsloth over a 10-teacher × 6-language grid. |
| [finetune_student_model_unsloth_ddp.sh](jobs/finetune_student_model_unsloth_ddp.sh) | Same as above but launched with `torchrun` DDP for larger base models (e.g., Olmo-3-32B). |
| [data_scale_effect.sh](jobs/data_scale_effect.sh) | Finetune students across data scales (1k–50k samples) for 3 languages to study the effect of dataset size. |
| [translation_effect.sh](jobs/translation_effect.sh) | Finetune students on NLLB-translated datasets to compare translation strategies against direct generation. |
| [eval_base_model.sh](jobs/eval_base_model.sh) | Run Lighteval on the base pretrained anchor model (Olmo-3-1025-7B) — the lower bound for PGR. |
| [eval_ref_model.sh](jobs/eval_ref_model.sh) | Run Lighteval on the reference instruct anchor model (Olmo-3-7B-Instruct-SFT) — the upper bound for PGR. |
| [eval_teacher_models.sh](jobs/eval_teacher_models.sh) | Run Lighteval on all candidate teacher models across the multilingual task suite. |
| [eval_student_models.sh](jobs/eval_student_models.sh) | Run Lighteval on every student revision of `msde-sft-dev` matching a search string. |
| [eval_tagalog_models.sh](jobs/eval_tagalog_models.sh) | Run Lighteval on the Tagalog/Cebuano-specific task suite for Tagalog student revisions. |
| [upload_best_models.sh](jobs/upload_best_models.sh) | Export the best per-language student revisions to public `Polyglot-*` HuggingFace repos. |
| [sync_isambard.sh](jobs/sync_isambard.sh) | Reinstall `torch`/`vllm`/`lighteval` on the Isambard compute node (CUDA wheels are unavailable on the login node). |