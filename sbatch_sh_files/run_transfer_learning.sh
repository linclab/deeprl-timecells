#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=ddc_to_dnms_data
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/timing_pretrained_data-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/timing_pretrained_data-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#model_seed_list=(104 106 109 110 114) #102 too  # DNMS dim-2, 6 out of 49 good seeds
model_seed_list=(107 125 124 101 135) #102 too  # DDC 9 out of 9 good seeds
task_seeds=(1 2 3 4 5 6 7 8 9 10)

model_seed_idx=$((($SLURM_ARRAY_TASK_ID - 1) / 10))
task_seed_idx=$((($SLURM_ARRAY_TASK_ID - 1) % 10))

model_seed=${model_seed_list[$model_seed_idx]}
task_seed=${task_seeds[$task_seed_idx]}

#load_model_path="mem_40_lstm_128_5e-05/seed_${model_seed}_epi199999.pt"  # DNMS dim-2 models
#load_model_path="lstm_128_1e-05/seed_${model_seed}_epi149999.pt"  # DDC models

#load_model_path="model_${model_seed}/seed_${task_seed}_epi149999.pt" # collect data on DNMS->DDC model
load_model_path="model_${model_seed}/seed_${task_seed}_epi199999.pt" # collect data on DDC-> DNMS model

#python expts/run_int_discrim_1d.py --model_seed $model_seed --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --lr 0.00001 --seed $task_seed --hidden_type 'lstm' --record_data True

python expts/run_tunl_1d.py --model_seed $model_seed --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $task_seed --env_type 'mem' --hidden_type 'lstm' --record_data True




#echo "Training DNMS model with model seed: $model_seed and DDC task seed: $task_seed"
#echo "Training DDC model with model seed: $model_seed and DMNS task seed: $task_seed"
#echo "Training DDC model with model seed: $model_seed and DNMS NoMem task seed: $task_seed"

#python expts/run_int_discrim_1d.py --n_total_episodes 150000 --save_ckpt_per_episodes 50000 --load_model_path $load_model_path --save_ckpts True --n_neurons 128 --lr 0.00001 --seed $task_seed --hidden_type 'lstm' --save_performance_fig True
#python expts/run_tunl_1d.py --n_total_episodes 200000 --save_ckpt_per_episodes 40000 --save_ckpts True --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $task_seed --env_type 'mem' --hidden_type 'lstm' --save_performance_fig True
#python expts/run_tunl_1d_nomem.py --n_total_episodes 200000 --save_ckpt_per_episodes 40000 --save_ckpts True --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $task_seed --env_type 'nomem' --hidden_type 'lstm' --save_performance_fig True
