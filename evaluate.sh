#!/bin/bash
#SBATCH -J EvalMemorization
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=eval_mem_%j.out
#SBATCH --error=eval_mem_%j.err
# ------------------------------------------------------------------
# 0. PATHS
# ------------------------------------------------------------------
THESIS_DIR="/home/jef08min/Thesis"
EVAL_SCRIPT="${THESIS_DIR}/blipevaluation.py"
MODEL_PATH="${THESIS_DIR}/finetuned_mem_analysis/ckpt_epoch_10"
# ------------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------------
module purge
module load python/3.10
module load anaconda3/latest
source $ANACONDA_HOME/etc/profile.d/conda.sh

echo "Activating environment..."
conda activate finetune

# ------------------------------------------------------------------
# 2. CHECK GPU
# ------------------------------------------------------------------
echo "Checking GPU visibility..."
nvidia-smi
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"

# ------------------------------------------------------------------
# 3. VERIFY FILES
# ------------------------------------------------------------------
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: Python script not found at $EVAL_SCRIPT"
    echo "Please save the python code from the previous chat as 'memorization_evaluation.py'"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "WARNING: Model folder not found at $MODEL_PATH"
    echo "Please check the MODEL_PATH variable in this script."
    # We exit here because running without a model will fail
    exit 1
fi

# ------------------------------------------------------------------
# 4. RUN EVALUATION
# ------------------------------------------------------------------
cd "$THESIS_DIR"
echo "Starting evaluation logic..."
echo "Model: $MODEL_PATH"

# Run the python script
# Note: We removed --batch_size from arguments because the new script 
# handles batching internally, but you can add it back if you modified the parser.
python "$EVAL_SCRIPT" \
    --model_path "$MODEL_PATH"

# ------------------------------------------------------------------
# 5. PRINT RESULTS
# ------------------------------------------------------------------
RESULTS_FILE="final_memorization_metrics.json"

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "===================================="
    echo "       MEMORIZATION RESULTS         "
    echo "===================================="
    cat "$RESULTS_FILE"
    echo ""
    echo "===================================="
else
    echo "Evaluation finished, but $RESULTS_FILE was not found."
    echo "Check the python logs above for errors."
fi

conda deactivate