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
