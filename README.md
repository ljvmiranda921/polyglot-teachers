<img src="assets/ltl_logo2.svg" height="70" align="right" /><img src="assets/cambridge_logo.png" height="70" align="right" />

## Polyglot Teachers: Evaluating Language Models for Multilingual Synthetic Data Generation 

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
git submodule update --init --recursive --depth 1
uv sync --dev
# When training on TPUs and developing models via tunix
# uv sync --extra tpu
# When doing evaluations via lighteval
# uv sync --extra eval
source .venv/bin/activate
```

Isambard is a bit different because the login node doesn't have a GPU (if you sync normally, it will install the CPU versions of pytorch which will mess up your virtual environment). Instead you should run these commands:

```sh
uv sync \
    --no-install-package triton \
    --no-install-package torch \
    --no-install-package torchaudio \
    --no-install-package torchvision \
    --no-install-package vllm \
    --no-install-package llama-cpp-python \
    --no-install-package ctranslate2 
sbatch experiments/slurm_submit.isambard experiments/jobs/sync_isambard.sh
```

For more information on how to use this codebase, please refer to the [documentation](DOCUMENTATION.md).
For information on running experiments on the cluster, see the [experiment documentation](experiments/README.md).
