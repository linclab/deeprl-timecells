#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=ddc_analysis
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing/tunl1d_pretrained_analysis-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing/tunl1d_pretrained_analysis-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#seed_list=(104 102 101 106 105 108 111 109 107 113 110 112 103 115 116 117 125 121 129 123 124 128 126 114 118 120 119 127 122 144 142 146 145 143 141 150 140 130 136 137 131 138 139 149 147 148)
#seed_list=(101 104 102 109 111 107 105 106 125 139 137 143 146 150 144 142 145 149)
#seed_list=(102 107 125 124 101 135 140 130 145)

#seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

# Define your list of seeds for the models and task seeds
seed_list=(104 106 109 110 114)
task_seeds=(1 2 3 4 5 6 7 8 9 10)

# Calculate the index for selecting the model seed and task seed
model_seed_idx=$((($SLURM_ARRAY_TASK_ID - 1) / 10))
task_seed_idx=$((($SLURM_ARRAY_TASK_ID - 1) % 10))

# Select the model seed and task seed based on the calculated indices
model_seed=${seed_list[$model_seed_idx]}
task_seed=${task_seeds[$task_seed_idx]}

#python analysis/fig_2_tunl1d_single_seed.py --seed $seed
python analysis/fig_2_timing_single_seed.py --seed $task_seed --model_seed $model_seed
