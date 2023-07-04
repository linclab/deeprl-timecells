#!/bin/bash
#SBATCH --array=0-50
#SBATCH --job-name=tunl1d_training
#SBATCH --output=training-%A_%a.out
#SBATCH --error=training-%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --mail-type=END,FAIL

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 100))

python expts/run_tunl_1d.py --n_total_episodes 200000 --save_ckpt_per_episodes 40000 --save_ckpts True --load_model_path 'None' --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'mem' --hidden_type 'lstm' --save_performance_fig True