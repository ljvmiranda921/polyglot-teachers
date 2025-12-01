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
source .venv/bin/activate
```

For more information on how to use this codebase, please refer to the [documentation](DOCUMENTATION.md).
For information on running experiments on the cluster, see the [experiment documentation](experiments/README.md).
