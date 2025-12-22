#!/bin/bash
# Local execution script for computing intrinsic metrics (non-SLURM)

MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "CohereLabs/aya-expanse-32b"
    "cohere-command-a"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "google/gemma-3-4b-it"
    "gpt-4o-mini-2024-07-18"
    "ibm-granite/granite-4.0-1b"
    "ibm-granite/granite-4.0-micro"
#    "mistralai/Mistral-Small-24B-Instruct-2501"
)

LANGUAGES=(ar cs de es id ja)

# Create metrics directory if it doesn't exist
mkdir -p ./metrics

for MODEL in "${MODELS[@]}"; do
    for LANGUAGE in "${LANGUAGES[@]}"; do
        echo "========================================="
        echo "Computing intrinsic metrics for model: ${MODEL} and language: ${LANGUAGE}"
        echo "========================================="

        INPUT_FILTER='{"model": "'"$MODEL"'"}'
        OUTPUT_PATH="./data/csd3/msde-S1-${LANGUAGE}_${MODEL//\//__}_intrinsic_metrics.json"

        python -m scripts.get_intrinsic_metrics \
            --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
            --metrics length \
            --output_path "$OUTPUT_PATH" \
            --input_dataset_filter "$INPUT_FILTER" \
            --apply_subsampling

        if [ $? -ne 0 ]; then
            echo "Error processing model: ${MODEL}, language: ${LANGUAGE}"
        else
            echo "Successfully processed model: ${MODEL}, language: ${LANGUAGE}"
        fi
        echo ""
    done
done

echo "All combinations processed!"
