import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards)+1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode-1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode-1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards

# ====== extract behavioural data and plot performance =====
# load data
conditions = ["R10.0_P1.0", "R10.0_P0.5", "R5.0_P1.0", "R5.0_P0.5"]
mean_perf = {}
std_perf = {}
for condition in conditions:
    data_dir = f'/network/scratch/l/lindongy/timecell/training/td_incentive/lstm_N128_1e-05_mc_len7_{condition}'
    seed_list = np.arange(101, 111)
    first_half_performance_array = np.zeros((len(seed_list), 30000))
    second_half_performance_array = np.zeros((len(seed_list), 30000))
    for i, seed in enumerate(seed_list):
        if "1.0" in condition:
            if os.path.exists(os.path.join(data_dir, f'seed_{seed}_total_30000episodes_performance_data.npz')):
                first_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_epi0_to_30000episodes_performance_data.npz'),
                                          allow_pickle=True)
                second_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_total_30000episodes_performance_data.npz'),
                                           allow_pickle=True)
                #['stim', 'epi_nav_reward', 'epi_incentive_reward', 'ideal_nav_reward', 'p_losses', 'v_losses']
                first_half_performance_array[i,:] = first_half_data['epi_nav_reward']
                second_half_performance_array[i,:] = second_half_data['epi_nav_reward']
            else:
                print(f'condition {condition}: seed {seed} not found')
                first_half_performance_array[i,:] = np.nan
                second_half_performance_array[i,:] = np.nan
        else:
            if os.path.exists(os.path.join(data_dir, f'seed_{seed}_epi29999_to_59999_performance_data.npz')):
                first_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_epi0_to_30000episodes_performance_data.npz'),
                                          allow_pickle=True)
                second_half_data = np.load(os.path.join(data_dir, f'seed_{seed}_epi29999_to_59999_performance_data.npz'),
                                           allow_pickle=True)
                #['stim', 'epi_nav_reward', 'epi_incentive_reward', 'ideal_nav_reward', 'p_losses', 'v_losses']
                first_half_performance_array[i,:] = first_half_data['epi_nav_reward']
                second_half_performance_array[i,:] = second_half_data['epi_nav_reward']
            else:
                print(f'condition {condition}: seed {seed} not found')
                first_half_performance_array[i,:] = np.nan
                second_half_performance_array[i,:] = np.nan

    performance_array = np.concatenate((first_half_performance_array, second_half_performance_array), axis=1)

    # plot performance with mean and std
    mean_perf[condition] = bin_rewards(np.nanmean(performance_array, axis=0), window_size=100)
    std_perf[condition] = bin_rewards(np.nanstd(performance_array, axis=0), window_size=100)

plt.figure(figsize=(8,6))
for condition in conditions:
    plt.plot(mean_perf[condition], label=condition, alpha=0.5)
    plt.fill_between(np.arange(60000), mean_perf[condition]-std_perf[condition], mean_perf[condition]+std_perf[condition], alpha=0.1)
plt.xlabel('Episode')
plt.ylabel('Navigation reward')
plt.ylim([-200, 10])
plt.legend()
plt.title("Incentive task performance, MC agents")
plt.savefig('/network/scratch/l/lindongy/timecell/figures/incentive/performance_mc_all.svg')
plt.savefig('/network/scratch/l/lindongy/timecell/figures/incentive/performance_mc_all.png')


