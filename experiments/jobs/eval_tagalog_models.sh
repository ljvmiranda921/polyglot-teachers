#!/bin/bash
TASKS=(
    "belebele_ceb_mcf"
    "cebuaner_ceb_mcf"
    "readability_ceb_mcf"
    "dengue_filipino_fil"
    "firecs_fil_mcf"
    "global_mmlu_cs_tgl_mcf"
    "include_tgl_mcf"
    "kalahi_tgl_mcf"
    "newsphnli_fil_mcf"
    "ntrex128_fil"
    "sib200_tgl_mcf"
    "sib200_ceb_mcf"
    "stingraybench_semantic_appropriateness_tgl_mcf"
    "stingraybench_correctness_tgl_mcf"
    "tatoeba_ceb"
    "tatoeba_tgl"
    "tico19_tgl"
    "tlunifiedner_tgl_mcf"
    "universalner_ceb_mcf"
    "universalner_tgl_mcf"
)

BASE_MODEL="ljvmiranda921/msde-sft-dev"

REVISIONS=(
    "20260116T095252-msde-google_gemma-3-4b-pt-lora-4bit-tgl_10k-Public-tl"
    "20260116T112300-msde-google_gemma-3-4b-pt-lora-4bit-tgl_10k-GPT-4om"
    "20260116T172611-msde-google_gemma-3-4b-pt-lora-4bit-tgl_10k-AyaExp"
    "20260116T173116-msde-google_gemma-3-4b-pt-lora-4bit-tgl_10k-Gemma3"
    "20260116T173212-msde-google_gemma-3-4b-pt-lora-4bit-tgl_25k-Gemma3"
    "20260116T213453-msde-google_gemma-3-12b-pt-lora-4bit-tgl_25k-Gemma3-Big"
    "20260117T125437-msde-google_gemma-3-27b-pt-lora-4bit-tgl_25k-Gemma3-Ultra"
)

# Print revision mapping for array job submission
echo "=== Available Revisions ==="
echo "To run all revisions, use: sbatch --array=0-$((${#REVISIONS[@]}-1)) ..."
for i in "${!REVISIONS[@]}"; do
    echo "  Array index $i: ${REVISIONS[$i]}"
done
echo "==========================="

REVISION=${REVISIONS[$SLURM_ARRAY_TASK_ID]}
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Selected revision: ${REVISION}"
echo "Evaluating model: ${BASE_MODEL} (revision: ${REVISION})"

for TASK in "${TASKS[@]}"; do
    echo "Evaluating task: ${TASK}"
    lighteval vllm "model_name=${BASE_MODEL},revision=${REVISION},tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_length=8192,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.1}" "${TASK}" \
        --custom-tasks lighteval/src/lighteval/tasks/multilingual/tasks/filipino.py \
        --output-dir lighteval-results \
        --results-path-template '{output_dir}/{org}___{model}' \
        --public-run \
        --results-org "ljvmiranda921" \
        --push-to-hub \
        --save-details

    echo "Sleeping for 30 seconds before next task..."
    sleep 30
done 
