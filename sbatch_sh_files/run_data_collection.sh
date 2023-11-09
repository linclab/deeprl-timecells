#!/bin/bash
#SBATCH --array=0-14
#SBATCH --job-name=sep_ac_data
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/sep_ac_data-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/sep_ac_data-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

#seed_list=(104 102 101 106 105 108 111 109 107 113 110 112 103 115 116 117 125 121 129 123 124 128 126 114 118 120 119 127 122 144 142 146 145 143 141 150 140 130 136 137 131 138 139 149 147 148)
#seed_list=(101 104 102 109 111 107 105 106 125 139 137 143 146 150 144 142 145 149)
#seed_list=(101 102 103 105 104 106 108 107 109 111 110 112 113 114 115 116 117 118 119 120 121 123 126 127 128 129 131 130 134 135 133 132 136 137 138 139 140 143 142 124 141 145 125 146 144 147 148 149 150)
#seed_list=(101 102 103 104 105 106 109 108 107 112 111 114 116 117 118 119 110 113 121 120)
seed_list=(542 544 678 766 805 783 788 1057 1089 1119 1227 1229 1282 1299)
epi_list=(399999 399999 399999 399999 319999 399999 399999 399999 399999 279999 399999 399999 399999 399999)
seed=${seed_list[$SLURM_ARRAY_TASK_ID]}
epi=${epi_list[$SLURM_ARRAY_TASK_ID]}
#seed=$((SLURM_ARRAY_TASK_ID + 100))
#load_model_path="nomem_40_lstm_128_5e-05/seed_${seed}_epi199999.pt"
load_model_path="mem_40_lstm_128_0.0001/seed_${seed}_epi${epi}.pt"

python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.0001 --seed $seed --env_type mem --hidden_type lstm --record_data True
#python expts/run_tunl_1d_nomem.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path $load_model_path --n_neurons 128 --len_delay 40 --lr 0.00005 --seed $seed --env_type 'nomem' --hidden_type lstm --record_data True
python analysis/fig_2_tunl1d_single_seed.py --seed $seed --epi $epi
