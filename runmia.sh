#!/bin/bash
#SBATCH -J MIA_FullAttack
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=mia_full_%j.out
#SBATCH --error=mia_full_%j.err

# ------------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------------
THESIS_DIR="/home/jef08min/Thesis"

# The Python script you just saved
MIA_SCRIPT="${THESIS_DIR}/mia.py"

# Path to your BEST checkpoint (Epoch 10)
MODEL_PATH="${THESIS_DIR}/finetuned_mem_analysis/ckpt_epoch_10"

# Batch size for the attack (32 is usually safe for A100/A6000)
# If you get OOM (Out of Memory), reduce this to 16 or 8
BATCH_SIZE=32

# ------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ------------------------------------------------------------------
module purge
module load python/3.10
module load anaconda3/latest
source $ANACONDA_HOME/etc/profile.d/conda.sh

echo "Activating 'finetune' environment..."
conda activate finetune

# ------------------------------------------------------------------
# 2. CHECKS
# ------------------------------------------------------------------
echo "Checking GPU..."
nvidia-smi

if [ ! -f "$MIA_SCRIPT" ]; then
    echo "ERROR: Python script not found at $MIA_SCRIPT"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found at $MODEL_PATH"
    exit 1
fi

# ------------------------------------------------------------------
# 3. RUN THE ATTACK
# ------------------------------------------------------------------
echo "========================================================"
echo "STARTING FULL DATASET MEMBERSHIP INFERENCE ATTACK"
echo "Model: $MODEL_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "========================================================"

python "$MIA_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

# ------------------------------------------------------------------
# 4. FINISH
# ------------------------------------------------------------------
echo ""
echo "========================================================"
if [ -f "${MODEL_PATH}/mia_results_full.csv" ]; then
    echo "SUCCESS: Results saved to ${MODEL_PATH}/mia_results_full.csv"
    echo "You can now download this CSV to plot distributions."
else
    echo "WARNING: CSV file was not generated. Check logs above."
fi
echo "========================================================"

conda deactivate