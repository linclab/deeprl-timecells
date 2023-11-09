#!/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=incentive_td
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl2d/incentive_td-%A.%a.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl2d/incentive_td-%A.%a.err
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1  # 2g.20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

seed=$((SLURM_ARRAY_TASK_ID + 110))
incentive_mag=5.0
incentive_prob=1.0
load_model_path="lstm_N128_1e-05_td_truncate2000_len7_R${incentive_mag}_P${incentive_prob}/seed_${seed}_epi29999.pt"

python expts/run_incentive.py --n_total_episodes 30000 --save_ckpt_per_episodes 5000 --load_model_path None --n_neurons 128 --lr 0.00001 --seed $seed --algo td --truncate_step 2000 --len_edge 7 --poke_reward 1 --incentive_mag $incentive_mag --incentive_prob $incentive_prob --save_ckpts True

