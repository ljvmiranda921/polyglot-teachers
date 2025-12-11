# Running on Isambard using Singularity

If you're running on an HPC cluster that uses Singularity, you can build and use the provided container:

```sh
# Base container (no extras)
singularity build --fakeroot mtep.sif singularity/mtep.def

# Or with eval extra for lighteval
singularity build --fakeroot mtep_eval.sif singularity/mtep_eval.def
```

Once the container is built on the cluster, you can run these scripts:

```sh
# Run as interactive shell
srun -N 1 --gpus 1 --pty singularity shell --nv mtep.sif

# Run directly (use uv run to activate the venv)
singularity exec --nv \
    --bind data:/app/data \
    --bind outputs:/app/outputs \
    --env HF_TOKEN=$HF_TOKEN \
    mtep.sif \
    uv run python scripts/get_intrinsic_metrics.py --model-name meta-llama/Llama-3.2-1B-Instruct

# Run as a batch job (see experiments/*)
sbatch experiments/slurm_submit_sif.isambard experiments/jobs/get_intrinsic_metrics_sif.sh
```