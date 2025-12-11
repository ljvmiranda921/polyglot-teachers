#!/bin/bash
#SBATCH --job-name=mtep
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multilingual Teacher Evaluation Project - Singularity Container Job Script
# This script runs your Python code inside the mtep.sif Singularity container

set -e

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Configuration
CONTAINER_IMAGE="mtep.sif"
SCRIPT_PATH="scripts/your_script.py"  # Change this to your script
BIND_PATHS="data:/app/data,outputs:/app/outputs,metrics:/app/metrics,models:/app/models"  # Adjust as needed

# Create logs directory if it doesn't exist
mkdir -p logs data outputs metrics models

srun singularity exec \
    --nv \
    --bind "$BIND_PATHS" \
    --env HF_TOKEN="$HF_TOKEN" \
    "$CONTAINER_IMAGE" \
    python "$SCRIPT_PATH" "$@"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
