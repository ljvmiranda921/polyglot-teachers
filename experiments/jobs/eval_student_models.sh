#!/bin/bash
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

BASE_MODEL="ljvmiranda921/msde-sft-dev"
SEARCH_STR="Olmo"

# Get all revisions matching the search string
REVISIONS=($(python scripts/utils/get_model_rev.py --hf_model_id "${BASE_MODEL}" --search_str="${SEARCH_STR}"))

REVISION=${REVISIONS[$SLURM_ARRAY_TASK_ID]}
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Selected revision: ${REVISION}"
echo "Evaluating model: ${BASE_MODEL} (revision: ${REVISION})"

for TASK in "${TASKS[@]}"; do
    echo "Evaluating task: ${TASK}"
    lighteval vllm "model_name=${BASE_MODEL},revision=${REVISION},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=4096,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
        --custom-tasks scripts/lighteval_tasks.py \
        --output-dir lighteval-results \
        --results-path-template '{output_dir}/{org}___{model}' \
        --no-public-run \
        --results-org "ljvmiranda921" \
        --push-to-hub \
        --save-details

    echo "Sleeping for 30 seconds before next task..."
    sleep 30
done 
