#!/bin/bash

# Run those with all languages available first
TASKS=(
    "global_mmlu_lite:de"
    "global_mmlu_lite:es"
    "global_mmlu_lite:ja"
    "mrewardbench_mcf:de"
    "mrewardbench_mcf:es"
    "mrewardbench_mcf:ja"
    "mgsm_custom:de|5"
    "mgsm_custom:es|5"
    "mgsm_custom:ja|5"
    "global_mmlu_lite:ar"
    "mrewardbench_mcf:ar"
    "global_mmlu_lite:id"
    "mrewardbench_mcf:id"
    "mrewardbench_mcf:cs"
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID % ${#TASKS[@]}]}
MODEL="allenai/Olmo-3-1025-7B"

echo "Evaluating model: ${MODEL} on task: ${TASK}"

# Reference for gsm8k setup: https://github.com/huggingface/lighteval/issues/686
lighteval vllm --help
lighteval vllm "model_name=${MODEL},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks scripts/lighteval_tasks.py \
    --output-dir lighteval-results \
    --results-path-template '{output_dir}/{org}___{model}' \
    --no-public-run \
    --results-org "ljvmiranda921" \
    --push-to-hub \
    --save-details 