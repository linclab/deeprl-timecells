#!/bin/bash
#SBATCH --array=0-46
#SBATCH --job-name=tunl1d_lesion
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/lesion-%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/lesion-%A_%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed_list=(101 102 104 105 106 108 112 113 118 119 120 126 129 130 132 135 136 137 138 140 144 145 154 165 167 168 172 176 184 192 194 200 202 204 207 209 213 227 233 234 237 238 240 245 247 248)
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}
model_path=$"mem_40_lstm_128_0.0001/seed_${seed}_epi199999.pt"
python expts/run_lesion_tunl1d.py --expt_type rehydration --lesion_side total --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
#python expts/run_lesion_tunl1d.py --expt_type lesion --lesion_side total --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
#python expts/run_lesion_tunl1d.py --expt_type rehydration --lesion_side left --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
#python expts/run_lesion_tunl1d.py --expt_type lesion --lesion_side left --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
#python expts/run_lesion_tunl1d.py --expt_type rehydration --lesion_side right --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
#python expts/run_lesion_tunl1d.py --expt_type lesion --lesion_side right --load_model_path $model_path --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 128 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99
