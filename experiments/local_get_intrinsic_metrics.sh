
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
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

LANGUAGES=(ar cs de es id ja)


METRIC_PARAMS='reward_model::{"model_name": "http://localhost:8080/v1", "provider":"openai_server","language":"'"$LANGUAGE"'"}'
INPUT_FILTER='{"model": "'"$MODEL"'"}'
OUTPUT_PATH="metrics/msde-S1-${LANGUAGE}_${MODEL//\//__}_intrinsic_metrics.json"
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --metrics reward_model \
    --output_path "$OUTPUT_PATH" \
    --metric_params "$METRIC_PARAMS" \
    --input_dataset_filter "$INPUT_FILTER" \
    --apply_subsampling