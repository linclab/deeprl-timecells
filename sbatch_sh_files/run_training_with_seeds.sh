#!/bin/bash
#SBATCH --job-name=tunl2d_seed
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl2d/seed_slurm-%j.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl2d/seed_slurm-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1  # 2g:20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0

source $HOME/testenv/bin/activate


# Run 2d experiment with selected HParam on 5 seeds
for seed in {2..5}
do python expts/run_tunl_2d.py --n_total_episodes 60000 --save_ckpt_per_episodes 10000 --load_model_path 'None' --n_neurons 512 --len_delay 40 --lr 0.00001 --seed $seed --env_type 'nomem'
done
