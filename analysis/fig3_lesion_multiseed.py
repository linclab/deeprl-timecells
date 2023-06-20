import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")
import argparse


# Load lesion results
# argparse and print args: expt, expt_type, lesion_side
parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, default='timing', help='timing, tunl1d, or tunl1dnomem')
parser.add_argument('--expt_type', type=str, default='lesion', help='lesion or rehydration')
parser.add_argument('--lesion_side', type=str, default='total', help='total, left, or right')
args = parser.parse_args()
expt = args.expt
expt_type = args.expt_type
lesion_side = args.lesion_side
print(f'Experiment: {expt}; Experiment type: {expt_type}; Lesion side: {lesion_side}')

if expt == 'timing':
    main_dir = '/network/scratch/l/lindongy/timecell/figures/lesion/timing1d/lstm_128_1e-05'
elif expt == 'tunl1d':
    main_dir = '/network/scratch/l/lindongy/timecell/figures/lesion/tunl1d_og/mem_40_lstm_128_0.0001'
elif expt == 'tunl1dnomem':
    main_dir = '/network/scratch/l/lindongy/timecell/figures/lesion/tunl1d_nomem/nomem_40_lstm_128_5e-05'
else:
    raise ValueError("expt must be timing, tunl1d, or tunl1dnomem")

# Loop through directories to grep seeds
seeds = []
for dir in os.listdir(main_dir):
    if os.path.isdir(os.path.join(main_dir, dir)):
        seeds.append(dir)
seeds = sorted(seeds)

# intialize arrays
postlesion_perf_random = []
postlesion_perf_ramp = []
postlesion_perf_time = []
if expt_type == 'rehydration':
    mean_kl_div_random = []
    mean_kl_div_ramp = []
    mean_kl_div_time = []

# Loop through seeds
for i, seed in enumerate(seeds):
    agent_str = f'{seed}_149999_100_99.0'
    results_dir = os.path.join(main_dir, agent_str, expt_type, lesion_side)
    if expt_type == 'lesion':
        if os.path.exists(os.path.join(results_dir, 'epi100_shuff50_idx5_5_128_lesion_results.npz')):
            results = np.load(os.path.join(results_dir, 'epi100_shuff50_idx5_5_128_lesion_results.npz'), allow_pickle=True)
        else:
            continue
    elif expt_type == 'rehydration':
        if os.path.exists(os.path.join(results_dir, 'epi100_shuff50_idx5_5_128_rehydration_results.npz')):
            results = np.load(os.path.join(results_dir, 'epi100_shuff50_idx5_5_128_rehydration_results.npz'), allow_pickle=True)
        else:
            continue
    else:
        raise ValueError('Invalid experiment type')

    postlesion_perf_random.append(results['postlesion_perf'][0])
    postlesion_perf_ramp.append(results['postlesion_perf'][1])
    postlesion_perf_time.append(results['postlesion_perf'][2])
    if expt_type == 'rehydration':
        mean_kl_div_random.append(results['mean_kl_div'][0])
        mean_kl_div_ramp.append(results['mean_kl_div'][1])
        mean_kl_div_time.append(results['mean_kl_div'][2])

num_good_seeds = len(postlesion_perf_random)
print(f'Total number of seeds: {len(seeds)}; number of good seeds: {num_good_seeds}')

postlesion_perf_time = np.vstack(postlesion_perf_time)  # (num_good_seeds*50, 25)
postlesion_perf_ramp = np.vstack(postlesion_perf_ramp)
postlesion_perf_random = np.vstack(postlesion_perf_random)
if expt_type == 'rehydration':
    mean_kl_div_time = np.vstack(mean_kl_div_time)
    mean_kl_div_ramp = np.vstack(mean_kl_div_ramp)
    mean_kl_div_random = np.vstack(mean_kl_div_random)

n_lesion = np.arange(5, 128, 5)

# Plot
postlesion_perf_time_mean = np.mean(postlesion_perf_time, axis=0)
postlesion_perf_time_std = np.std(postlesion_perf_time, axis=0)
postlesion_perf_ramp_mean = np.mean(postlesion_perf_ramp, axis=0)
postlesion_perf_ramp_std = np.std(postlesion_perf_ramp, axis=0)
postlesion_perf_random_mean = np.mean(postlesion_perf_random, axis=0)
postlesion_perf_random_std = np.std(postlesion_perf_random, axis=0)
if expt_type == 'rehydration':
    mean_kl_div_random_mean = np.mean(mean_kl_div_random, axis=0)
    mean_kl_div_random_std = np.std(mean_kl_div_random, axis=0)
    mean_kl_div_ramp_mean = np.mean(mean_kl_div_ramp, axis=0)
    mean_kl_div_ramp_std = np.std(mean_kl_div_ramp, axis=0)
    mean_kl_div_time_mean = np.mean(mean_kl_div_time, axis=0)
    mean_kl_div_time_std = np.std(mean_kl_div_time, axis=0)

save_dir = '/network/scratch/l/lindongy/timecell/figures/lesion'
fig, ax1 = plt.subplots()
fig.suptitle(f'{expt}_{expt_type}_{lesion_side}')
ax1.plot(n_lesion, postlesion_perf_random_mean, color='gray', label=f'Random {"lesion" if expt_type  == "lesion" else "silencing"}')
ax1.fill_between(n_lesion,
                 postlesion_perf_random_mean-postlesion_perf_random_std,
                 postlesion_perf_random_mean+postlesion_perf_random_std, color='lightgray', alpha=0.2)
ax1.plot(n_lesion, postlesion_perf_ramp_mean, color='blue', label=f'Ramping cell {"lesion" if expt_type  == "lesion" else "silencing"}')
ax1.fill_between(n_lesion,
                    postlesion_perf_ramp_mean-postlesion_perf_ramp_std,
                    postlesion_perf_ramp_mean+postlesion_perf_ramp_std, color='lightblue', alpha=0.2)
ax1.plot(n_lesion, postlesion_perf_time_mean, color='red', label=f'Time cell {"lesion" if expt_type  == "lesion" else "silencing"}')
ax1.fill_between(n_lesion,
                    postlesion_perf_time_mean-postlesion_perf_time_std,
                    postlesion_perf_time_mean+postlesion_perf_time_std, color='pink', alpha=0.2)
ax1.hlines(y=0.5, xmin=0, xmax=1, transform=ax1.get_yaxis_transform(), linestyles='dashed', colors='gray')
ax1.set_xlabel(f'Number of neurons {"lesioned" if expt_type  == "lesion" else "silenced"}')
ax1.set_ylabel('% Correct')
ax1.set_ylim(0,1)
ax1.legend()
ax1.title(f"pooled across {num_good_seeds} seeds")
#plt.show()
fig.savefig(os.path.join(save_dir, f'{expt}_{expt_type}_{lesion_side}.svg'))
fig.savefig(os.path.join(save_dir, f'{expt}_{expt_type}_{lesion_side}.png'))


if expt_type  == "rehydration":
    fig, ax1 = plt.subplots()
    fig.suptitle(f'{expt}_{expt_type}_{lesion_side}')
    ax1.plot(n_lesion, mean_kl_div_random_mean, color='gray', label=f'Random {"lesion" if expt_type  == "lesion" else "silencing"}')
    ax1.fill_between(n_lesion,
                        mean_kl_div_random_mean-mean_kl_div_random_std,
                        mean_kl_div_random_mean+mean_kl_div_random_std, color='lightgray', alpha=0.2)
    ax1.plot(n_lesion, mean_kl_div_ramp_mean, color='blue', label=f'Ramping cell {"lesion" if expt_type  == "lesion" else "silencing"}')
    ax1.fill_between(n_lesion,
                        mean_kl_div_ramp_mean-mean_kl_div_ramp_std,
                        mean_kl_div_ramp_mean+mean_kl_div_ramp_std, color='lightblue', alpha=0.2)
    ax1.plot(n_lesion, mean_kl_div_time_mean, color='red', label=f'Time cell {"lesion" if expt_type  == "lesion" else "silencing"}')
    ax1.fill_between(n_lesion,
                        mean_kl_div_time_mean-mean_kl_div_time_std,
                        mean_kl_div_time_mean+mean_kl_div_time_std, color='pink', alpha=0.2)
    ax1.set_xlabel(f'Number of neurons {"lesioned" if expt_type  == "lesion" else "silenced"}')
    ax1.set_ylabel('KL divergence between $\pi$ and $\pi_{new}$')
    ax1.legend()
    ax1.title(f"pooled across {num_good_seeds} seeds")
    #plt.show()
    fig.savefig(os.path.join(save_dir, f'{expt}_{expt_type}_{lesion_side}_kl.svg'))
    fig.savefig(os.path.join(save_dir, f'{expt}_{expt_type}_{lesion_side}_kl.png'))

print("Done!")

