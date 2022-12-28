#!/bin/bash
#SBATCH --array=0-15%15
#SBATCH --job-name=tunl1d_inp4_hparam
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_inp4/slurm-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_inp4/slurm-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1  # 2g:20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0
source $HOME/testenv/bin/activate

len_delay_arr=(40)
env_type_arr=('mem')
hidden_type_arr=('lstm')
lr_arr=(0.00001 0.00005 0.0001 0.0005 0.001)
n_neurons_arr=(128 256)
#seed_arr=(1 2 3 4 5)

lenLD=${#len_delay_arr[@]}
lenET=${#env_type_arr[@]}
lenHT=${#hidden_type_arr[@]}
lenLR=${#lr_arr[@]}
lenN=${#n_neurons_arr[@]}


lenMul12=$((lenLR*lenN))
lenMul123=$((lenLR*lenN*lenHT))
lenMul1234=$((lenLR*lenN*lenHT*lenET))

ldidx=$((SLURM_ARRAY_TASK_ID/lenMul1234))
idx1234=$((SLURM_ARRAY_TASK_ID%lenMul1234))
etidx=$((idx1234/lenMul123))
idx123=$((idx1234%lenMul123))
htidx=$((idx123/lenMul12))
idx12=$((idx123%lenMul12))
lridx=$((idx12/lenN))
nidx=$((idx12%lenN))

len_delay=${len_delay_arr[$ldidx]}
env_type=${env_type_arr[$etidx]}
hidden_type=${hidden_type_arr[$htidx]}
lr=${lr_arr[$lridx]}
n_neurons=${n_neurons_arr[$nidx]}

load_model_path="$env_type $len_delay $hidden_type $n_neurons $lr/seed_1_epi999999.pt"
load_model_path=$(echo $load_model_path | sed 's/ /_/g')

# Run 1d experiment
python expts/run_tunl_1d.py --n_total_episodes 1000000 --save_ckpt_per_episodes 100000 --load_model_path $load_model_path --save_ckpts True --n_neurons $n_neurons --len_delay $len_delay --lr $lr --seed 1 --env_type $env_type --hidden_type $hidden_type
# Note: if want record_data to be False, don't pass anything. Otherwise it will parse at True.


# Save data from 1d experiment checkpoint
#python expts/run_tunl_1d.py --n_total_episodes 500 --save_ckpt_per_episodes 250 --record_data True --load_model_path '2022-12-25_21-07/lstm_mem_512_Epi499.pt' --n_neurons 512 --len_delay 40 --lr 0.0001 --seed 1 --env_type 'mem' --hidden_type 'lstm'
# Note: if want save_ckpts to be False, don't pass anything. Otherwise it will parse at True.

# Run 2d experiment
# python expts/run_tunl_2d.py --n_total_episodes 60000 --save_ckpt_per_episodes 10000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --len_delay $len_delay --lr $lr --seed 1 --env_type $env_type --hidden_type $hidden_type
# Note: if want record_data to be False, don't pass anything. Otherwise it will parse at True.
