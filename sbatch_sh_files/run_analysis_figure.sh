#!/bin/bash
#SBATCH --array=0-87
#SBATCH --job-name=timing1d_figures
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing/figures-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing/figures-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

num_units_list=(128 128 256 256 512 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512)

lr_list=(5e-06 1e-05 5e-06 1e-05 5e-06 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 1e-05 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06 5e-06)

dropout_prob_list=(0.0 0.0 0.0 0.0 0.0 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.5 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.1 0.1 0.1 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.5 0.5 0.5 0.1 0.1 0.1 0.25 0.25 0.25 0.5 0.1 0.1 0.25 0.25 0.5 0.5 0.5 0.1 0.1 0.25 0.25 0.5 0.5 0.1 0.1 0.25 0.25 0.25 0.5 0.1 0.1 0.25 0.25 0.5 0.5 0.5)

dropout_type_list=(0 0 0 0 0 2 2 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2 2 2 2 2 3 3 3 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 2 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3 3 3 3 3)

seed_list=(1 1 1 1 1 1 2 3 1 2 3 1 2 1 2 3 2 3 3 2 3 1 2 3 1 3 1 2 3 1 2 3 1 2 3 1 2 1 2 3 1 2 3 1 2 3 1 2 3 1 3 1 2 3 1 2 3 1 2 3 1 2 3 2 3 1 2 3 1 2 2 3 1 3 1 2 1 2 3 3 2 3 2 3 1 2 3)

num_units=${num_units_list[$SLURM_ARRAY_TASK_ID]}
lr=${lr_list[$SLURM_ARRAY_TASK_ID]}
dropout_prob=${dropout_prob_list[$SLURM_ARRAY_TASK_ID]}
dropout_type=${dropout_type_list[$SLURM_ARRAY_TASK_ID]}
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

if [[ $dropout_type -eq 0 ]]; then
  data_dir="lstm_${num_units}_${lr}"
else
  data_dir="lstm_${num_units}_${lr}_p${dropout_prob}_${dropout_type}"
fi  

python analysis/analysis_script_int_discrim_1d.py --main_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/data/timing1d --main_save_dir /home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/analysis_results/timing1d --n_shuffle 100 --percentile 99.999 --data_dir $data_dir --seed $seed --episode 149999

