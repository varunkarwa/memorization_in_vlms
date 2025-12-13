#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH -J FinetuneVLM
#SBATCH --mail-type=END,FAIL
#SBATCH -n 4               # must equal number of GPUs
#SBATCH -N 1               # single node ONLY

# -------------------------------
# 1. Modules & Conda
# -------------------------------
module purge
module load python/3.10
module load anaconda3/latest
module load cuda/12.8
source $ANACONDA_HOME/etc/profile.d/conda.sh

ENV_NAME=finetune_vlm

if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda env $ENV_NAME ..."
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

# -------------------------------
# 2. Install dependencies once
# -------------------------------
CACHE_DIR=$HOME/.cache/torch_wheels
mkdir -p $CACHE_DIR

if [ ! -f "$HOME/.cache/${ENV_NAME}_installed_ok" ]; then
    echo "Installing dependencies..."

    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt

    pip uninstall -y torch torchvision torchaudio

    # Install correct PyTorch nightly for Blackwell
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --extra-index-url https://download.pytorch.org/whl/nightly \
        -t $CACHE_DIR

    pip install $CACHE_DIR/torch-*.whl \
        $CACHE_DIR/torchvision-*.whl \
        $CACHE_DIR/torchaudio-*.whl \
        --no-index --find-links $CACHE_DIR

    pip install bitsandbytes==0.43.3

    touch "$HOME/.cache/${ENV_NAME}_installed_ok"
    echo "Dependencies installed."
fi

# -------------------------------
# 3. Verify GPUs
# -------------------------------
python - <<'EOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU[{i}]:", torch.cuda.get_device_name(i))
EOF

# -------------------------------
# 4. Proper DDP environment
# -------------------------------
# MASTER_ADDR = first node hostname
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Use a random port to avoid EADDRINUSE
export MASTER_PORT=$((10000 + RANDOM % 50000))

# WORLD_SIZE = total tasks (1 task per GPU)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE  RANK=$RANK  LOCAL_RANK=$LOCAL_RANK"

# -------------------------------
# 5. Launch training
# -------------------------------
echo "Starting training with 4 DDP processes..."

# Important: srun must NOT override env vars → use --export=ALL
srun --export=ALL \
     --cpu-bind=v \
     --accel-bind=g \
     python webqa_finetuning.py

echo "Done."
conda deactivate
