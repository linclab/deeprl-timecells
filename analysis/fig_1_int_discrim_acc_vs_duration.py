import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")
from analysis.utils_int_discrim import plot_performance


data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/timing/lstm_128_1e-05'
save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_2/timing1d'

# loop through data_dir and find files that have file name 'lstm_128_1e-05_seed_{seed}_epi149999.pt_data.npz'
good_seeds = []
for file_name in os.listdir(data_dir):
    if 'lstm_128_1e-05_seed' in file_name:
        pt = re.search('lstm_128_1e-05_seed_(\d+)_epi149999.pt_data.npz', file_name)
        seed = int(pt[1])
        if seed == 102 or seed == 189:
            continue
        good_seeds.append(seed)
print(f'Found {len(good_seeds)} good seeds: {good_seeds}')

acc_stim1_longer_arr = np.zeros((len(good_seeds), 6))  # [5, 10, 15, 20, 25, 30]
acc_stim2_longer_arr = np.zeros((len(good_seeds), 6))  # [5, 10, 15, 20, 25, 30]
for i, seed in enumerate(good_seeds):
    seed_save_dir = os.path.join(save_dir, f'seed_{seed}')
    if not os.path.exists(seed_save_dir):
        os.makedirs(seed_save_dir, exist_ok=True)
    data = np.load(os.path.join(data_dir, f'lstm_128_1e-05_seed_{seed}_epi149999.pt_data.npz'), allow_pickle=True)
    action_hist = data["action_hist"]
    correct_trials = data["correct_trial"]
    stim = data["stim"]
    stim1_resp = data["stim1_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    stim2_resp = data["stim2_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    delay_resp = data["delay_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    n_total_episodes = np.shape(stim)[0]
    acc_stim1_longer_arr[i], acc_stim2_longer_arr[i] = plot_performance(stim, action_hist,  title=f'seed_{seed}', save_dir=seed_save_dir, fig_type='curve', save=True)

acc_stim1_longer_mean = np.mean(acc_stim1_longer_arr, axis=0)
acc_stim1_longer_std = np.std(acc_stim1_longer_arr, axis=0)
acc_stim2_longer_mean = np.mean(acc_stim2_longer_arr, axis=0)
acc_stim2_longer_std = np.std(acc_stim2_longer_arr, axis=0)
duration_diff = [5, 10, 15, 20, 25, 30]
fig, ax = plt.subplots()
ax.plot(duration_diff, acc_stim1_longer_mean, label="stim1 - stim2", marker='o')
ax.plot(duration_diff, acc_stim2_longer_mean, label="stim2 - stim1", marker='o')
ax.fill_between(duration_diff, acc_stim1_longer_mean - acc_stim1_longer_std, acc_stim1_longer_mean + acc_stim1_longer_std, alpha=0.2)
ax.fill_between(duration_diff, acc_stim2_longer_mean - acc_stim2_longer_std, acc_stim2_longer_mean + acc_stim2_longer_std, alpha=0.2)
ax.set_xlabel("Difference in stimulus duration")
ax.set_ylabel("Accuracy")
ax.set_title("Task performance")
ax.legend(frameon=False)
plt.show()
plt.savefig(os.path.join(save_dir, 'acc_vs_duration.svg'))
