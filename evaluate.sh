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
EVAL_SCRIPT="${THESIS_DIR}/evaluationmemorization.py"
MODEL_PATH="${THESIS_DIR}/finetuned_docvqa/epoch3"

# ------------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------------
module purge
module load python/3.10
module load anaconda3/latest

source $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate finetune

# ------------------------------------------------------------------
# 2. INSTALL LATEST PYTORCH NIGHTLY + TORCHVISION
# ------------------------------------------------------------------
echo "Installing latest PyTorch nightly + torchvision..."

pip uninstall -y torch torchvision torchaudio timm 2>/dev/null || true

# Install LATEST nightly (no version pinning)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124

pip install timm

# ------------------------------------------------------------------
# 3. VERIFY TORCHVISION
# ------------------------------------------------------------------
python - <<'PY'
import torch, torchvision, timm
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"timm: {timm.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# ------------------------------------------------------------------
# 4. FIX MODEL LOADING IN SCRIPT
# ------------------------------------------------------------------
# Replace AutoModelForCausalLM → AutoModelForImageTextToText
if grep -q "AutoModelForCausalLM" "$EVAL_SCRIPT"; then
    echo "Fixing model class: AutoModelForCausalLM → AutoModelForImageTextToText"
    sed -i 's/AutoModelForCausalLM/AutoModelForImageTextToText/g' "$EVAL_SCRIPT"
fi

# Fix torch_dtype → dtype
if grep -q "torch_dtype=" "$EVAL_SCRIPT"; then
    echo "Fixing torch_dtype → dtype"
    sed -i 's/torch_dtype=/dtype=/g' "$EVAL_SCRIPT"
fi

# ------------------------------------------------------------------
# 5. Verify model
# ------------------------------------------------------------------
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    ls -la "${THESIS_DIR}/finetuned_docvqa/" || true
    exit 1
fi
echo "Model: $MODEL_PATH"

# ------------------------------------------------------------------
# 6. Run evaluation
# ------------------------------------------------------------------
cd "$THESIS_DIR"

echo "Starting evaluation..."
python "$EVAL_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --batch_size 8

# ------------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------------
RESULTS="${MODEL_PATH}/memorisation_results.json"
if [ -f "$RESULTS" ]; then
    echo ""
    echo "MEMORIZATION RESULTS"
    echo "===================================="
    cat "$RESULTS" | python -m json.tool
    echo "===================================="
else
    echo "Results missing: $RESULTS"
fi

conda deactivate