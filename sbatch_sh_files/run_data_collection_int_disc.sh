#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=timing_no_reset_data
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/timing/no_reset_data-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/timing/no_reset_data-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#seed_list=(101 102 103 104 105 109 110 111 112 113 114 115 116 117 118 119 120 124 125 126 127 134 144 145 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 181 183 185 188 189 191 192 193 195 196 197 199 200)

#seed_list=(101 102 103 104 105 109 110 111 112 113 115 116 117 118 119 120 124 125 126 127 134 144 145 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 174 175 176 177 178 179 181 183 185 188 189 191 192 193 195 196 197 200)  #Timing1d trained

#seed_list=(102 107 125 124 101 135 140 130 145)

#seed=${seed_list[$SLURM_ARRAY_TASK_ID]}

#python analysis/fig_2_timing_single_seed.py --seed $seed --untrained True

#python analysis/fig_2_timing_single_seed.py --seed $seed

seed=$((SLURM_ARRAY_TASK_ID + 100))
load_model_path="lstm_128_1e-05/seed_${seed}_epi149999.pt"

python expts/run_int_discrim_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --lr 1e-05 --seed $seed --hidden_type 'lstm' --record_data True
