#!/bin/bash
#SBATCH --array=0-50
#SBATCH --job-name=tunl1d_analysis
#SBATCH --output=analysis-%A_%a.out
#SBATCH --error=analysis-%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 100))

python analysis/fig_2_tunl1d_single_seed.py--seed $seed