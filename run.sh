#!/bin/bash
#SBATCH -t 72:00:00 # 4 hours
#SBATCH --gres=gpu:4 # request 4 GPUs
#SBATCH --mem=256G # 256 GB CPU RAM
#SBATCH --cpus-per-task=4
#SBATCH -J FinetuneVLM
#SBATCH --mail-type=END,FAIL
#SBATCH -n 4
#SBATCH -N 1

module purge
module load python/3.10 # or 3.9/3.11 if available
# Load Anaconda
module load anaconda3/latest
# Initialize conda
. $ANACONDA_HOME/etc/profile.d/conda.sh

# Activate your environment
conda activate finetune

if [ ! -f "$HOME/.cache/finetune_reqs_installed" ]; then
    echo "Installing Python packages..."
    pip install --upgrade pip
    # Install from requirements.txt (this includes a good stable torch with CUDA support)
    pip install -r requirements.txt

    # === OPTIONAL: If you need latest nightly for Blackwell/B200 fixes ===
    # Comment out the above pip install -r requirements.txt line for torch-related pkgs if using this
    # Current (Dec 2025) nightly uses cu128 or cu126; cu124 is deprecated/no longer built
    # pip uninstall -y torch torchvision torchaudio
    # pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    # (Replace cu128 with cu126 if your cluster driver supports it better)

    touch "$HOME/.cache/finetune_reqs_installed"
    echo "Dependencies installed."
else
    echo "Dependencies already installed. Skipping."
    pip install --upgrade "transformers>=4.36.0" "peft>=0.7.0" "accelerate>=0.25.0"
fi

# Quick verification that torch is installed and sees GPUs
python - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
EOF

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Each process sees only its own GPU (for multi-GPU)
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "=== SLURM Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Tasks: $SLURM_NTASKS"
echo "ProcID: $SLURM_PROCID"
echo "LocalID: $SLURM_LOCALID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "========================"

# Launch training
srun --cpu-bind=v --accel-bind=g \
     python debug.py

conda deactivate