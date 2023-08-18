import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

# DNMS to DDC
# Define helper functions
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
data_dir = '/network/scratch/l/lindongy/timecell/training/timing/tunl1d_pretrained/lstm_128_1e-05'
# look for folders that match "model_$model'
performance_by_model = {}
seed_by_model = {}
for file in os.listdir(data_dir):
    if re.match(f'model_\d+', file):
        performance_by_model[(file.split('_')[1])] = []
        seed_by_model[(file.split('_')[1])] = []
for model in performance_by_model.keys():
    print(f'Processing model {model}')
    subdir = os.path.join(data_dir, f'model_{model}')
    for file in os.listdir(subdir):
        if re.match(f'seed_\d+_total_150000episodes_performance_data.npz', file):
            seed_by_model[model].append(int(file.split('_')[1]))
            # stim = np.load(os.path.join(subdir, file), allow_pickle=True)['stim']
            # first_action = np.load(os.path.join(subdir, file), allow_pickle=True)['first_action']
            # performance array the same shape as stim, where stim+first_action is 1
            # performance_by_model[model].append((stim+first_action==1).astype(int))
            performance_by_model[model].append(np.load(os.path.join(subdir, file), allow_pickle=True)['correct_trial'])
    performance_by_model[model] = np.array(performance_by_model[model])
breakpoint()

# plot average and stf performance of each model
plt.figure(figsize=(8,6))
for model in performance_by_model.keys():
    mean_perf = bin_rewards(np.mean(performance_by_model[model], axis=0), window_size=1000)
    std_perf = bin_rewards(np.std(performance_by_model[model], axis=0), window_size=1000)
    plt.plot(mean_perf, label=f'Model {model} for DNMS')
    plt.fill_between(np.arange(150000), mean_perf-std_perf, mean_perf+std_perf, alpha=0.1)
plt.xlabel('Episode')
plt.ylabel('% Correct')
plt.ylim([0,1])
plt.legend()
plt.title("DNMS to DDC")
plt.savefig('/network/scratch/l/lindongy/timecell/figures/tunl1d_to_ddc_performance_.svg')
plt.savefig('/network/scratch/l/lindongy/timecell/figures/tunl1d_to_ddc_performance_.png')




