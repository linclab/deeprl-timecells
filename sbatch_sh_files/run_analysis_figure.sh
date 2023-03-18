#!/bin/bash
#SBATCH --array=0-8
#SBATCH --job-name=tunl1d_figures
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/figures-%j.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/figures-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

mem_list=("mem" "mem" "mem" "mem" "mem" "nomem" "nomem" "nomem")

num_units_list=(128 128 256 256 512 128 256 512)

lr_list=(0.0001 0.0001 0.0001 0.0001 0.0001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.001 0.001 0.001 0.001 0.001 0.0001 0.0001 0.0001 0.0001 0.0001)

dropout_prob_list=(0.0 0.0 0.5 0.25 0.1 0.0 0.0 0.0)

dropout_type_list=(0 0 2 2 2 0 0 0)

seed_list=(1 3 2 3 2 2 3 6)

episode_list=(199999 199999 199999 199999 199999 199999 199999 199999)

mem=${mem_list[$SLURM_ARRAY_TASK_ID]}
num_units=${num_units_list[$SLURM_ARRAY_TASK_ID]}
lr=${lr_list[$SLURM_ARRAY_TASK_ID]}
dropout_prob=${dropout_prob_list[$SLURM_ARRAY_TASK_ID]}
dropout_type=${dropout_type_list[$SLURM_ARRAY_TASK_ID]}
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}
episode=${episode_list[$SLURM_ARRAY_TASK_ID]}

if [[ "$dropout_type" -eq 0 || "$dropout_type" == "0" ]]; then
  model_path=${mem}"_40_lstm_${num_units}_${lr}"
else
  model_path=${mem}"_40_lstm_${num_units}_${lr}_p${dropout_prob}_${dropout_type}"
fi

python analysis/analysis_script_tunl1d.py --main_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/data/tunl1d --main_save_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/analysis_figures/tunl1d --n_shuffle 1000 --percentile 99.999 --data_dir $model_path --seed $seed --episode $episode
