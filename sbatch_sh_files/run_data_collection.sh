#!/bin/bash
#SBATCH --array=0-50
#SBATCH --job-name=tunl1d_data
#SBATCH --output=data-%A_%a.out
#SBATCH --error=data-%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 100))
model_path=$"mem_40_lstm_128_0.0001/seed_${seed}_epi199999.pt"

python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --record_data True --load_model_path $model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'mem' --hidden_type 'lstm'