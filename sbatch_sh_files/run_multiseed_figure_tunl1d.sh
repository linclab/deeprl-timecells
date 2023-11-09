#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=tunl_no_reset_analysis
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/no_reset_analysis-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/no_reset_analysis-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#seed_list=(101 102 104 105 106 108 112 113 118 119 120 126 129 130 132 135 136 137 138 140 144 145 154 165 167 168 172 176 184 192 194 200 202 204 207 209 213 227 233 234 237 238 240 245 247 248) # TUNL1d Mem trained

#seed_list=(101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 153 154 155) # TUNL2d Mem trained

#seed_list=(101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 141 142 143 144 146 148 149 150 151 153 154 155)  # TUNL2d Mem untraind

#seed_list=(101 102 103 104 106 107 109 111 112 113 115 132 147 150 161 162 166 167 171 172 173 174 175 176 177 178 179 180 181 182 183 184 189 190 191 192 194 195 196 197 198 199 200) # TUNL2d NoMem trained or untrained

#seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

seed=$((SLURM_ARRAY_TASK_ID + 100))
#python analysis/fig_2_timing_single_seed.py --seed $seed

python analysis/fig_2_tunl1d_single_seed.py --seed $seed

#python analysis/fig_7_tunl2d_single_seed.py --seed $seed

#python analysis/fig_7_tunl2d_single_seed.py --seed $seed --untrained True

#python analysis/fig_7_tunl2d_nomem_single_seed.py --seed $seed

#python analysis/fig_7_tunl2d_nomem_single_seed.py --seed $seed --untrained True
