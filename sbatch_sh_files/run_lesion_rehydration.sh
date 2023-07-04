#!/bin/bash
#SBATCH --array=0-50
#SBATCH --job-name=tunl1d_lesion
#SBATCH --output=lesion-%A_%a.out
#SBATCH --error=lesion-%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 100))
model_path=$"mem_40_lstm_128_0.0001/seed_${seed}_epi199999.pt"
python expts/run_lesion_tunl1d.py --expt_type rehydration --lesion_side total --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
