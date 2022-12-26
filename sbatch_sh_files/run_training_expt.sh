#!/bin/bash
#SBATCH --job-name=tunl
#SBATCH --output=sbatch_out/slurm-%j.out
#SBATCH --error=sbatch_err/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0

source $HOME/testenv/bin/activate

tmp_date=/network/scratch/l/lindongy/timecell/training/$(date '+%Y-%m-%d_%H-%M')/
mkdir $tmp_date

# Run 1d experiment
python expts/run_tunl_1d.py --save_dir $tmp_date --n_total_episodes 500 --save_ckpt_per_episodes 250 --record_data False\
 --load_model_path 'None' --save_ckpts True --n_neurons 512 --len_delay 40 --lr 0.0001 --seed 1 --env_type 'mem' --hidden_type 'lstm'

# Save data from 1d experiment checkpoint
