#!/bin/bash

#SBATCH -t 1                # time limit set to 15 minutes
#SBATCH --mem=4096           # 4G of memory reserved
#SBATCH -J PythonJob         # job name
#SBATCH --mail-type=END      # email sent at the end of the job
#SBATCH -n 1                 # 1 processor
#SBATCH -N 1                 # 1 node

# # module avail python     # see available versions
# module load python/3.10 # or 3.9/3.11 if available

# python -m venv ~/venvs/finetune
# source ~/venvs/finetune/bin/activate

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# # pip3 install peft --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
# pip3 install git+https://github.com/huggingface/peft.git
# # pip3 install datasets --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org

# # accelerate config

module purge

# Load Anaconda
module load anaconda3/latest

# Initialize conda in this job
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda create -n finetune python=3.10 -y

# Activate your environment (replace "finetune" with your env name)
conda activate finetune

# Run your script
pip3 install -r requirements.txt

# Deactivate conda after job
conda deactivate
# pip install transformers datasets accelerate peft bitsandbytes
