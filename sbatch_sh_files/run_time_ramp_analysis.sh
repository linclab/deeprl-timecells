#!/bin/bash
#SBATCH --array=0-45
#SBATCH --job-name=tunl2d_time_ramp
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl2d/time_ramp-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl2d/time_ramp-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1  # 2g.20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

mem_list=("mem" "mem" "mem" "mem" "nomem" "nomem" "nomem" "nomem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem" "mem")

num_units_list=(512 256 512 256 512 256 512 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512)

lr_list=(5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06)

dropout_prob_list=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.5 0.1 0.1 0.1 0.25 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.5)

dropout_type_list=(0 0 0 0 0 0 0 0 2 3 3 3 3 3 3 3 3 2 3 3 3 3 3 2 2 2 2 3 3 3 3 3 3 3 3 3 2 3 3 3 3 3 3 3 3)

seed_list=(1 1 1 1 1 1 1 1 1 1 3 1 2 3 1 2 3 1 1 2 1 3 3 1 2 3 1 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3)

mem=${mem_list[$SLURM_ARRAY_TASK_ID]}
num_units=${num_units_list[$SLURM_ARRAY_TASK_ID]}
lr=${lr_list[$SLURM_ARRAY_TASK_ID]}
dropout_prob=${dropout_prob_list[$SLURM_ARRAY_TASK_ID]}
dropout_type=${dropout_type_list[$SLURM_ARRAY_TASK_ID]}
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

if [[ "$dropout_type" -eq 0 ]]; then
  model_path="${mem}_40_lstm_${num_units}_${lr}"
else
  model_path="${mem}_40_lstm_${num_units}_${lr}_p${dropout_prob}_${dropout_type}"
fi

python analysis/time_ramp_ident_tunl2d.py --main_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/data/tunl2d --data_dir $model_path --main_save_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/analysis_results/tunl2d --seed $seed --episode 59999 --n_shuffle 1000 --percentile 99.999

