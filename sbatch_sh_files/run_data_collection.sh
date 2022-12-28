#!/bin/bash
#SBATCH --job-name=tunl2d_data
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl2d/data_slurm-%j.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl2d/data_slurm-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1  # 2g:20gb for tunl1d, rtx8000 for tunl2d
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.
module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.0

source $HOME/testenv/bin/activate

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_lstm_512_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 512 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'mem' --hidden_type 'lstm'

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'nomem_40_lstm_512_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 512 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'nomem' --hidden_type 'lstm'

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_lstm_256_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 256 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'mem' --hidden_type 'lstm'

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'nomem_40_lstm_256_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 256 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'nomem' --hidden_type 'lstm'

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_lstm_128_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 128 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'mem' --hidden_type 'lstm'

python expts/run_tunl_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'nomem_40_lstm_128_1e-05/seed_1_epi59999.pt' --record_data True --n_neurons 128 --len_delay 40 --lr 0.00001  --seed 1 --env_type 'nomem' --hidden_type 'lstm'
