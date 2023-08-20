import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")


# ====== extract behavioural data and plot performance =====
# load data
data_dir = '/network/scratch/l/lindongy/timecell/training/tunl1d_og/no_reset/mem_40_lstm_128_0.0001'
seed_list = []
for file in os.listdir(data_dir):
    if re.match(f'seed_\d+_total_200000episodes_performance_data.npz', file):
        seed_list.append(int(file.split('_')[1]))
performance_array = np.zeros((len(seed_list), 200000))
for i, seed in enumerate(seed_list):
    data = np.load(os.path.join(data_dir, f'seed_{seed}_total_200000episodes_performance_data.npz'), allow_pickle=True)
    stim = data['stim']
    first_action = data['first_action']
    performance_array[i,:] = stim+first_action==1
    #performance_array[i,:] = data['correct_trial']


# plot performance with mean and std
mean_perf = np.mean(performance_array, axis=0)
std_perf = np.std(performance_array, axis=0)

plt.figure(figsize=(8,6))
plt.plot(mean_perf, label='128-unit LSTM')
plt.fill_between(np.arange(150000), mean_perf-std_perf, mean_perf+std_perf, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('% Correct')
plt.ylim([0,1])
plt.legend()
plt.title("DDC no reset")
plt.savefig('/network/scratch/l/lindongy/timecell/figures/timing_no_reset_performance.svg')
plt.savefig('/network/scratch/l/lindongy/timecell/figures/timing_no_reset_performance.png')

