#!/bin/bash
#SBATCH --array=0-45
#SBATCH --job-name=no_reset_timing
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing/no_reset-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing/no_reset-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1  # 2g:20gb for 1d, rtx8000 for 2d
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

lr_arr=(0.00001)
n_neurons_arr=(128)
wd_arr=(0)
seed_arr=(106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150)

lenLR=${#lr_arr[@]}
lenN=${#n_neurons_arr[@]}
lenWD=${#wd_arr[@]}
lenS=${#seed_arr[@]}

lenMul12=$((lenWD*lenS))
lenMul123=$((lenWD*lenS*lenN))

lridx=$((SLURM_ARRAY_TASK_ID/lenMul123))
idx123=$((SLURM_ARRAY_TASK_ID%lenMul123))
nidx=$((idx123/lenMul12))
idx12=$((idx123%lenMul12))
wdidx=$((idx12/lenS))
sidx=$((idx12%lenS))

lr=${lr_arr[$lridx]}
n_neurons=${n_neurons_arr[$nidx]}
wd=${wd_arr[$wdidx]}
seed=${seed_arr[$sidx]}

#load_model_path="lstm_${n_neurons}_1e-06/seed_${seed}_epi209999.pt"

# Run 1D interval discrimination experiment
python expts/run_int_discrim_1d.py --n_total_episodes 150000 --save_ckpt_per_episodes 50000 --load_model_path 'None' --save_ckpts True --n_neurons 128 --lr $lr --seed $seed --hidden_type 'lstm'


# Run 2D interval discrimination experiment
#python expts/run_int_discrim_2d.py --n_total_episodes 120000 --save_ckpt_per_episodes 30000 --load_model_path $load_model_path --save_ckpts True --n_neurons $n_neurons --lr $lr --seed $seed --hidden_type 'lstm' --save_performance_fig True
