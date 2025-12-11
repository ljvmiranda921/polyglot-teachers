# Multilingual Teacher Evaluation Project

In this project, we ask the question: "what makes a good multilingual teacher for synthetic data generation?"
Specifically, we perform a comprehensive analysis of several language models and evaluate their data quality as teacher models, *(intrinsic)* and the performance gain of the resulting student model on some benchmarks *(extrinsic)*.

<p align="center">
<img src="/assets/distillation_workflow.png" alt="Distillation Workflow" width="700" style="display: block; margin: 0 auto;"/>
<br/>
<i>Overview of the Polyglot Score and how it fits into the distillation workflow.</i>
</p>

## Setup & Installation

Make sure that you have `uv` in your system (see [download instructions](https://docs.astral.sh/uv/getting-started/installation/)).
To install all dependencies, run the following commands: 

```sh
git submodule update --init --recursive
uv sync --dev
# When training on TPUs and developing models via tunix
# uv sync --extra tpu
# When doing evaluations via lighteval
# uv sync --extra eval
source .venv/bin/activate
```

For more information on how to use this codebase, please refer to the [documentation](DOCUMENTATION.md).
For information on running experiments on the cluster, see the [experiment documentation](experiments/README.md).

### Running on Isambard using Singularity

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
