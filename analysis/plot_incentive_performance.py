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
data_dir = '/network/scratch/l/lindongy/timecell/training/td_incentive/lstm_N128_1e-05_mc_len7_R10.0_P1.0'
seed_list = np.arange(101, 111)
first_half_performance_array = np.zeros((len(seed_list), 30000))
second_half_performance_array = np.zeros((len(seed_list), 30000))
for i, seed in enumerate(seed_list):
    if os.path.exists(os.path.join(data_dir, f'seed_{seed}_total_30000episodes_performance_data.npz')):
        first_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_epi0_to_30000episodes_performance_data.npz'),
                                  allow_pickle=True)
        second_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_total_30000episodes_performance_data.npz'),
                                   allow_pickle=True)
        #['stim', 'epi_nav_reward', 'epi_incentive_reward', 'ideal_nav_reward', 'p_losses', 'v_losses']
        first_half_performance_array[i,:] = first_half_data['epi_nav_reward']
        second_half_performance_array[i,:] = second_half_data['epi_nav_reward']
    else:
        print(f'seed {seed} not found')
        first_half_performance_array[i,:] = np.nan
        second_half_performance_array[i,:] = np.nan
performance_array = np.concatenate((first_half_performance_array, second_half_performance_array), axis=1)

# plot performance with mean and std
mean_perf = np.nanmean(performance_array, axis=0)
std_perf = np.nanstd(performance_array, axis=0)

plt.figure(figsize=(8,6))
plt.plot(mean_perf, label='128-unit LSTM')
plt.fill_between(np.arange(60000), mean_perf-std_perf, mean_perf+std_perf, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('# of steps')
plt.legend()
plt.title("Incentive task performance, MC agents")
plt.savefig('/network/scratch/l/lindongy/timecell/figures/incentive/performance_mc.svg')

