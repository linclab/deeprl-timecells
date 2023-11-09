#!/bin/bash
#SBATCH --array=0-108%50
#SBATCH --job-name=timing2d_dropout
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing2d/dropout-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing2d/dropout-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1  # 2g.20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

len_delay_arr=(2 3) #dropout_type
#len_delay_arr=(40)
#env_type_arr=('mem')
env_type_arr=(0.1 0.25 0.5) # p_dropout
hidden_type_arr=(1 2 3) # seed
lr_arr=(0.000005 0.000001)
n_neurons_arr=(128 256 512)

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

python expts/run_int_discrim_2d.py --n_total_episodes 400000 --save_ckpt_per_episodes 40000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --lr $lr --seed $hidden_type --hidden_type 'lstm' --p_dropout $env_type --dropout_type $len_delay --save_performance_fig True

#python expts/run_tunl_1d.py --n_total_episodes 200000 --save_ckpt_per_episodes 40000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --len_delay 40 --lr $lr --p_dropout $env_type --dropout_type $len_delay --seed $hidden_type --env_type 'mem' --hidden_type 'lstm' --save_performance_fig True

#python expts/run_tunl_2d.py --n_total_episodes 60000 --save_ckpt_per_episodes 10000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --len_delay 40 --lr $lr --p_dropout $env_type --dropout_type $len_delay --seed $hidden_type --env_type 'mem' --hidden_type 'lstm' --save_performance_fig True
