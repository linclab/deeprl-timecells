import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

# Loop through sbatch output files and identify good seeds that have completed 199999 episodes
sbatch_out_dir = '/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og'
good_seeds = []
slurm_ids = ['3309998']
for slurm_id in slurm_ids:
    for job_array_id in range(1,51):
        file_name = f'nomem-{slurm_id}.{job_array_id}.out'
        file_path = os.path.join(sbatch_out_dir, file_name)
        with open(file_path, 'r') as f:
            # first line is: {'n_total_episodes': 200000, 'save_ckpt_per_episodes': 40000, 'record_data': False, 'load_model_path': 'None', 'save_ckpts': True, 'n_neurons': 128, 'len_delay': 40, 'lr': 0.0001, 'weight_decay': 0.0, 'seed': 110, 'env_type': 'mem', 'hidden_type': 'lstm', 'save_performance_fig': True, 'p_dropout': 0.0, 'dropout_type': None}
            # find 'seed' in the first line
            first_line = f.readline()
            pt = re.search("'seed': (\d+)", first_line)
            seed = int(pt[1])
            # See if '199999' is in the file
            f.seek(0)
            if '199999' in f.read():
                good_seeds.append(seed)
                print(f'Found good seed {seed} in {file_name}')
print(f'Found {len(good_seeds)} good seeds: {good_seeds}')

# ====== extract behavioural data and plot performance =====
# load data
data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl1d_og/nomem_40_lstm_128_5e-05'
performance_array = np.zeros((len(good_seeds), 200000))
for i, seed in enumerate(good_seeds):
    data = np.load(os.path.join(data_dir, f'nomem_40_lstm_128_5e-05_seed_{seed}_epi199999.pt_data.npz'), allow_pickle=True)
    performance_array[i,:] = data['nomem_perf']
    print(f'Loaded seed {seed}')

# plot performance with mean and std
mean_perf = np.mean(performance_array, axis=0)
std_perf = np.std(performance_array, axis=0)

plt.figure(figsize=(8,6))
plt.plot(mean_perf, label='128-unit LSTM')
plt.fill_between(np.arange(200000), mean_perf-std_perf, mean_perf+std_perf, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('% Correct')
plt.ylim([0,1])
plt.legend()
plt.title("NoMem DNMS")
plt.savefig('/network/scratch/l/lindongy/timecell/figures/tunl1d_nomem_performance.svg')

