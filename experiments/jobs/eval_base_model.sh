#!/bin/bash

# Run those with all languages available first
TASKS=(
    "global_mmlu_lite:de|3"
    "global_mmlu_lite:es|3"
    "global_mmlu_lite:ja|3"
    "mrewardbench_mcf:de|3"
    "mrewardbench_mcf:es|3"
    "mrewardbench_mcf:ja|3"
    "mgsm_custom:de|5"
    "mgsm_custom:es|5"
    "mgsm_custom:ja|5"
    "global_mmlu_lite:ar|3"
    "mrewardbench_mcf:ar|3"
    "global_mmlu_lite:id|3"
    "mrewardbench_mcf:id|3"
    "mrewardbench_mcf:cs|3"
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