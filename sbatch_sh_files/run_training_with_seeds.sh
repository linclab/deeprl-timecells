#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=nocue_nomem_dim2
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/nocue_nomem_dim2-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/nocue_nomem_dim2-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#seed=$((SLURM_ARRAY_TASK_ID + 100))

#load_model_path="nomem_40_lstm_128_5e-05/seed_${seed}_epi199999.pt"




seed_list=(102 107 125 124 101 135 140 130 145)

seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

load_model_path="lstm_128_1e-05/seed_${seed}_epi149999.pt"

python expts/run_tunl_1d_nomem.py --n_total_episodes 200000 --save_ckpt_per_episodes 40000 --save_ckpts True --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'nomem' --hidden_type 'lstm' --save_performance_fig True
#python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'None' --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'nomem' --hidden_type 'lstm' --record_data True

#python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'nomem' --hidden_type 'lstm' --record_data True

#python expts/run_int_discrim_1d.py --n_total_episodes 150000 --save_ckpt_per_episodes 50000 --load_model_path 'None' --save_ckpts True --n_neurons 128 --lr 0.00001 --seed $seed --hidden_type 'lstm' --save_performance_fig True

#python expts/run_tunl_2d.py --n_total_episodes 80000 --save_ckpt_per_episodes 20000 --save_ckpts True --load_model_path 'None' --n_neurons 256 --len_delay 40 --lr 0.000005 --seed $seed --env_type 'nomem' --hidden_type 'lstm'
