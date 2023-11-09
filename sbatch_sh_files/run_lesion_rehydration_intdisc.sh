#!/bin/bash
#SBATCH --array=0-1
#SBATCH --job-name=timing1d_rehydration
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing/rehydration-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing/rehydration-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

num_units_list=(128)
lr_list=(1e-05)
dropout_prob_list=(0.5)
dropout_type_list=(2)
seed_list=(1)

num_units=${num_units_list[$SLURM_ARRAY_TASK_ID]}
lr=${lr_list[$SLURM_ARRAY_TASK_ID]}
dropout_prob=${dropout_prob_list[$SLURM_ARRAY_TASK_ID]}
dropout_type=${dropout_type_list[$SLURM_ARRAY_TASK_ID]}
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

if [[ $dropout_type -eq 0 ]]; then
  data_dir="lstm_${num_units}_${lr}/seed_${seed}_epi149999.pt"
else
  data_dir="lstm_${num_units}_${lr}_p${dropout_prob}_${dropout_type}/seed_${seed}_epi149999.pt"
fi

python expts/run_lesion_int_discrim_1d.py --load_model_path $data_dir --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end $num_units --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 100 --ramp_time_percentile 99.999 --expt_type rehydration
