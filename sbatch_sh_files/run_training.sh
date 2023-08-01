#!/bin/bash
#SBATCH --job-name=tunl1d_nonneg
#SBATCH --output=/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og/nonneg_data-%j.out
#SBATCH --error=/network/scratch/l/lindongy/timecell/sbatch_err/tunl1d_og/nonneg_data-%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/a100_env/bin/activate

python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_256_5e-05/seed_1_epi199999.pt' --n_neurons 256 --len_delay 40 --lr 0.00005 --seed 1 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True
python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_128_0.0001/seed_3_epi159999.pt' --n_neurons 128 --len_delay 40 --lr 0.0001 --seed 3 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True
python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_512_5e-05/seed_3_epi79999.pt' --n_neurons 512 --len_delay 40 --lr 0.00005 --seed 3 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True
python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_128_0.0001/seed_1_epi79999.pt' --n_neurons 128 --len_delay 40 --lr 0.0001 --seed 1 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True
python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_256_0.0001/seed_1_epi119999.pt' --n_neurons 256 --len_delay 40 --lr 0.0001 --seed 1 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True
python expts/run_tunl_1d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'mem_40_rnn_128_5e-05/seed_1_epi159999.pt' --n_neurons 128 --len_delay 40 --lr 0.00005 --seed 1 --env_type 'mem' --hidden_type 'rnn' --record_data True --save_performance_fig True

#python expts/run_int_discrim_2d.py --n_total_episodes 5000 --save_ckpt_per_episodes 2500 --load_model_path 'lstm_512_1e-06/seed_1_epi299998.pt' --record_data True --n_neurons 512 --lr 0.000001 --seed 1 --hidden_type 'lstm' --save_performance_fig True

#python expts/run_lesion_tunl1d.py --load_model_path 'mem_40_lstm_256_0.0001_p0.5_2/seed_2_epi199999.pt' --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 256 --lesion_idx_step 5 --n_total_episodes 200 --n_ramp_time_shuffle 1000 --ramp_time_percentile 99.999 --expt_type lesion --save_net_and_data True

#python expts/run_lesion_tunl2d.py  --load_model_path 'mem_40_lstm_512_1e-05/seed_1_epi59999.pt' --num_shuffle 50 --lesion_idx_start 5 --lesion_idx_end 100 --lesion_idx_step 5 --n_total_episodes 500
