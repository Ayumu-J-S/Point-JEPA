#!/bin/bash
#SBATCH --job-name=classification_job   
#SBATCH --nodes=5                       
#SBATCH --cpus-per-task=3               
#SBATCH --mem=16G                       
#SBATCH --time=03:00:00                 
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=compute_canada/classification_%j.out  # Standard output and error log

# Load your Python module or any other modules
module load python/3.10

# If you're using a virtual environment, activate it
source ~/cc_ssl/bin/activate

export WANDB_API_KEY="$(cat ~/.wandb_cred)"
# Run your Python script with additional arguments
python -m pointjepa.tasks.classification fit -c configs/classification/modelnet40.yaml -c configs/wandb/pointjepa/classification_modelnet40.yaml $@
