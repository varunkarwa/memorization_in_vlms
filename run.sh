#!/bin/bash
#SBATCH -t 72:00:00          # 4 hours
#SBATCH --gres=gpu:4         # request 1 GPU
#SBATCH --mem=256G            # 125 GB CPU RAM
#SBATCH --cpus-per-task=4
#SBATCH -J FinetuneVLM
#SBATCH --mail-type=END,FAIL
#SBATCH -n 4
#SBATCH -N 1

# module avail python     # see available versions
module purge
module load python/3.10 # or 3.9/3.11 if available

# Load Anaconda
module load anaconda3/latest

# Initialize conda in this job
. $ANACONDA_HOME/etc/profile.d/conda.sh

#conda create -n finetune python=3.10 -y

# Activate your environment (replace "finetune" with your env name)
conda activate finetune

if [ ! -f "$HOME/.cache/finetune_reqs_installed" ]; then
    echo "Installing Python packages..."
    pip install --upgrade pip

    # Install from requirements.txt
    pip install -r requirements.txt

    # === CRITICAL: Install PyTorch Nightly for B200 (sm_100) ===
    pip uninstall -y torch torchvision torchaudio
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

    touch "$HOME/.cache/finetune_reqs_installed"
    echo "Dependencies installed."
else
    echo "Dependencies already installed. Skipping."
fi

# Make sure pip installs packages to your user space (avoids permission errors)
# pip3 install -r requirements.txt
export MASTER_ADDR=$(hostname -s)     # short hostname
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS       # 4
export RANK=$SLURM_PROCID             # 0,1,2,3
export LOCAL_RANK=$SLURM_LOCALID      # 0,1,2,3

# Ensure each process sees only its GPU
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug
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

# ==============================
# 4. Launch Training with srun
# ==============================
srun --cpu-bind=v --accel-bind=g \
     python finetuning.py

# Deactivate
conda deactivate
