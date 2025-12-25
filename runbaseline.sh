#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH -J BaselineEval
#SBATCH --mail-type=END,FAIL
#SBATCH --output=eval_mem_%j.out
#SBATCH --error=eval_mem_%j.err

module purge
module load python/3.10
module load anaconda3/latest
# Try loading CUDA module (Adjust version 11.8/12.1 based on your cluster availability)
module load cuda/11.8 || module load cuda/12.1 || echo "CUDA module not found, relying on system paths."

. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate finetune

echo "=== Dependency Check ==="
# 1. Check if PyTorch thinks it has a GPU
# We do this BEFORE running the main script to debug the environment
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" > cuda_check.log 2>&1

if grep -q "CUDA Available: False" cuda_check.log; then
    echo "ERROR: PyTorch cannot see the GPU. Attempting to reinstall PyTorch with CUDA support..."
    pip uninstall -y torch torchvision
    # Install PyTorch with CUDA 11.8 support (Standard for most clusters)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    # Also fix bitsandbytes which relies on CUDA
    pip install --upgrade "bitsandbytes>=0.41.0" "accelerate>=0.26.0" "transformers>=4.36.0"
else
    echo "SUCCESS: PyTorch sees the GPU."
fi

echo "=== Starting Baseline Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# CRITICAL FIX: Do NOT manually set CUDA_VISIBLE_DEVICES. 
# SLURM sets this automatically. Overwriting it breaks visibility.
# export CUDA_VISIBLE_DEVICES=0  <-- REMOVED THIS LINE

# Run the python script
python baselineevaluation.py \
    --model_id "Salesforce/blip2-opt-2.7b"

echo "Evaluation Complete. Results saved to baseline_results.json"
conda deactivate