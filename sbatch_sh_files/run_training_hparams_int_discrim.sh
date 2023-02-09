#!/bin/bash
#SBATCH --array=0-30%30
#SBATCH --job-name=timing_hparam
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing2d/slurm-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing2d/slurm-%A.%a.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1  # 2g:20gb for 1d, rtx8000 for 2d
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0

source $HOME/testenv/bin/activate

hidden_type_arr=('lstm' 'linear')
lr_arr=(0.000005 0.00001 0.00005 0.0001 0.0005)
n_neurons_arr=(128 256 512)

lenHT=${#hidden_type_arr[@]}
lenLR=${#lr_arr[@]}
lenN=${#n_neurons_arr[@]}

lenMul12=$((lenLR*lenN))

htidx=$((SLURM_ARRAY_TASK_ID/lenMul12))
idx12=$((SLURM_ARRAY_TASK_ID%lenMul12))
lridx=$((idx12/lenN))
nidx=$((idx12%lenN))

hidden_type=${hidden_type_arr[$htidx]}
lr=${lr_arr[$lridx]}
n_neurons=${n_neurons_arr[$nidx]}

# Run 1D interval discrimination experiment
python expts/run_int_discrim.py --n_total_episodes 150000 --save_ckpt_per_episodes 30000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --lr $lr --seed 1 --hidden_type $hidden_type --save_performance_fig True


# Run 2D interval discrimination experiment
python expts/run_int_discrim_2d.py --n_total_episodes 300000 --save_ckpt_per_episodes 30000 --load_model_path 'None' --save_ckpts True --n_neurons $n_neurons --lr $lr --seed 1 --hidden_type $hidden_type --save_performance_fig True