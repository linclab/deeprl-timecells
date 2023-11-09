#!/bin/bash
#SBATCH --array=1-100
#SBATCH --job-name=sep_ac_seeds
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/sep_ac-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/sep_ac-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1  # 2g.20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 700))

#load_model_path="mem_40_lstm_128_${lr}/seed_${len_delay}_epi199999.pt"

# Run 1d experiment
python expts/run_tunl_1d.py --n_total_episodes 400000 --save_ckpt_per_episodes 40000 --load_model_path None --save_ckpts True --n_neurons 128 --len_delay 40 --lr 0.0001 --seed $seed --env_type 'mem' --hidden_type 'lstm'
#python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr $lr --seed $len_delay --env_type 'nomem' --hidden_type 'lstm' --record_data True
# Note: if want record_data to be False, don't pass anything. Otherwise it will parse at True.

# Run 2d experiment
#python expts/run_tunl_2d.py --n_total_episodes 80000 --save_ckpt_per_episodes 20000 --load_model_path 'None' --save_ckpts True --n_neurons 256 --len_delay 40 --lr $lr --seed $len_delay --env_type 'nomem' --hidden_type 'lstm'
# Note: if want record_data to be False, don't pass anything. Otherwise it will parse at True.
