import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis.utils_analysis import time_decode_lin_reg, sort_resp, plot_sorted_averaged_resp
from analysis.utils_int_discrim import single_cell_temporal_tuning
from analysis.utils_mutual_info import joint_encoding_information_time_stimulus
from analysis.utils_time_ramp import ridge_to_background_varying_duration, lin_reg_ramping_varying_duration, \
    trial_reliability_vs_shuffle_score_varying_duration
from scipy.stats import kruskal
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")


def plot_time_decoding_across_seeds(t_test_dict, t_test_pred_dict, label):
    print("Plot time decoding across seeds...")
    len_delay = 20 if label == 'delay' else 40
    t_test_arr = t_test_pred_arr = np.zeros((len(t_test_dict), len_delay*400))
    for i, seed in enumerate(list(t_test_dict.keys())):
        t_test_arr[i,:] = np.array(list(t_test_dict[seed][label]))
        t_test_pred_arr[i,:] = np.array(list(t_test_pred_dict[seed][label]))
    t_test_arr = t_test_arr.flatten()
    t_test_pred_arr = t_test_pred_arr.flatten()
    mean_pred = np.zeros(len_delay)
    std_pred = np.zeros(len_delay)
    for t_elapsed in range(len_delay):
        mean_pred[t_elapsed] = np.mean(t_test_pred_arr[t_test_arr == t_elapsed])
        std_pred[t_elapsed] = np.std(t_test_pred_arr[t_test_arr == t_elapsed])# 1 x len_delay
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x=t_test_arr, y=t_test_pred_arr,s=1)
    ax.set_xlabel('Time since delay onset')
    ax.set_ylabel('Decoded time')
    ax.plot(np.arange(len_delay), np.arange(len_delay),color='k')
    ax.plot(np.arange(len_delay), mean_pred, color=utils_linclab_plot.LINCLAB_COLS['blue'])
    ax.fill_between(np.arange(len_delay), mean_pred-std_pred,  mean_pred+std_pred, color='skyblue',alpha=0.4)
    ax.set_xticks([0, len_delay])
    ax.set_xticklabels(['0', str(len_delay)])
    ax.set_yticks([0, len_delay])
    ax.set_yticklabels(['0', str(len_delay)])
    ax.set_title(f'Time decoding during {label} pooled across {len(t_test_dict)} seeds')
    ax.set_xlim((np.min(t_test_pred_arr),41))
    ax.set_ylim((np.min(t_test_pred_arr),np.max(t_test_pred_arr)))
    fig.savefig(os.path.join(save_dir, f'{label}_time_decoding_pooled_{len(t_test_dict)}seeds_{n_total_episodes}_{n_shuffle}_{percentile}.svg'), dpi=300)


def plot_mutual_information_across_seeds(info_dict):
    # info_dict[seed][stim_len] = n_sig_neurons x 3
    print("Plot mutual information across seeds...")
    info_arr = []
    for seed in info_dict.keys():
        for stim_len in info_dict[seed].keys():
            if len(info_dict[seed][stim_len])==0:
                continue
            info_arr.append(info_dict[seed][stim_len])
    info_arr = np.vstack(info_arr)
    n_total_neurons = np.shape(info_arr)[0]
    stats = cbook.boxplot_stats(info_arr, labels=['Stim x Time', r'$Stim x Rand(Time)$', r'$Time x Rand(Stim)$'], bootstrap=10000)
    for i in range(len(stats)):
        stats[i]['whislo'] = np.min(info_arr[:,i], axis=0)
        stats[i]['whishi'] = np.max(info_arr[:,i], axis=0)
    fig, axs = plt.subplots(1,1)
    fig.suptitle(f'Mutual Information pooled across {len(info_dict)} seeds')
    for i in range(n_total_neurons):
        plt.plot([1, 2, 3], info_arr[i,:], color="gray", lw=1)
    props = dict(color='indigo', linewidth=1.5)
    axs.bxp(stats, showfliers=False, boxprops=props,
            capprops=props, whiskerprops=props, medianprops=props)
    plt.ylabel("Mutual information (bits)", fontsize=19)
    #Run nonparametric test on unrmd, stimrmd, timermd pairwise, and print p-values
    print(f"Pooled_{len(info_dict)}seeds: Kruskal-Wallis test p-values:")
    unrmd, timermd, stimrmd = info_arr[:,0], info_arr[:,1], info_arr[:,2]
    print("Unrmd vs. Stimrmd: ", kruskal(unrmd, stimrmd)[1])
    print("Unrmd vs. Timermd: ", kruskal(unrmd, timermd)[1])
    print("Stimrmd vs. Timermd: ", kruskal(stimrmd, timermd)[1])
    plt.savefig(os.path.join(save_dir, f'joint_encoding_info_pooled_{len(info_dict)}seeds_{n_total_episodes}_{n_shuffle}_{percentile}.svg'))


def plot_count_time_and_ramping_cells(time_cell_ids, ramping_cell_ids):
    time_cell_counts = []
    ramping_cell_counts = []
    both_cell_counts = []
    neither_cell_counts = []
    for each_seed in time_cell_ids.keys():
        time_cell_counts.append(len(time_cell_ids[each_seed]['total']))
        ramping_cell_counts.append(len(ramping_cell_ids[each_seed]['total']))
        both_cell_counts.append(len(np.intersect1d(time_cell_ids[each_seed]['total'], ramping_cell_ids[each_seed]['total'])))
        not_time_cell = np.setdiff1d(np.arange(n_neurons), time_cell_ids[each_seed]['total'])
        not_ramping_cell = np.setdiff1d(np.arange(n_neurons), ramping_cell_ids[each_seed]['total'])
        neither_cell_counts.append(len(np.intersect1d(not_time_cell, not_ramping_cell)))
    # convert to numpy array
    time_cell_counts = np.array(time_cell_counts)
    ramping_cell_counts = np.array(ramping_cell_counts)
    both_cell_counts = np.array(both_cell_counts)
    neither_cell_counts = np.array(neither_cell_counts)
    # plot as bar plot with error bar
    fig, ax = plt.subplots()
    ax.bar(x=np.arange(4), height=[np.mean(time_cell_counts), np.mean(ramping_cell_counts), np.mean(both_cell_counts), np.mean(neither_cell_counts)], yerr=[np.std(time_cell_counts), np.std(ramping_cell_counts), np.std(both_cell_counts), np.std(neither_cell_counts)])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Time cells', 'Ramping cells', 'Both', 'Neither'])
    ax.set_title(f'Cell type count across {len(time_cell_ids)} seeds')
    ax.axhline(y=n_neurons, linestyle='--', color='k')
    plt.savefig(os.path.join(save_dir, f'cell_type_count_{len(time_cell_ids)}seeds_{n_total_episodes}_{n_shuffle}_{percentile}.svg'))


data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/timing/lstm_128_1e-05'
seed_list = []
for file in os.listdir(data_dir):
    if re.match(f'lstm_128_1e-05_seed_\d+_epi149999.pt_data.npz', file):
        seed_list.append(int(file.split('_')[4]))
seed_list = sorted(seed_list)
n_neurons = 128
n_shuffle = 100
percentile = 99
n_total_episodes = 5000


save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_2/timing1d'
os.makedirs(save_dir, exist_ok=True)
load = False

ramping_cell_ids = {}
time_cell_ids = {}
t_test_dict = {}
t_test_pred_dict = {}
info_dict = {}
for i_seed, each_seed in enumerate(seed_list):
    # load each seed's dicts from seed_save_dir
    seed_save_dir = os.path.join(save_dir, f'seed_{each_seed}')
    ramping_cell_ids[each_seed] = np.load(os.path.join(seed_save_dir, 'ramping_cell_ids_seed.npy'), allow_pickle=True).item()
    time_cell_ids[each_seed] = np.load(os.path.join(seed_save_dir, 'time_cell_ids_seed.npy'), allow_pickle=True).item()
    t_test_dict[each_seed] = np.load(os.path.join(seed_save_dir, 't_test_dict_seed.npy'), allow_pickle=True).item()
    t_test_pred_dict[each_seed] = np.load(os.path.join(seed_save_dir, 't_test_pred_dict_seed.npy'), allow_pickle=True).item()
    info_dict[each_seed] = np.load(os.path.join(seed_save_dir, 'info_dict_seed.npy'), allow_pickle=True).item()

plot_count_time_and_ramping_cells(time_cell_ids, ramping_cell_ids)

plot_mutual_information_across_seeds(info_dict)

for label in ['stimulus_1', 'stimulus_2', 'delay']:
    plot_time_decoding_across_seeds(t_test_dict, t_test_pred_dict, label)

# Save results
print("Save results..")
np.save(os.path.join(save_dir, f't_test_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_dict)
np.save(os.path.join(save_dir, f't_test_pred_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_pred_dict)
np.save(os.path.join(save_dir, f'info_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), info_dict)
np.save(os.path.join(save_dir, f'time_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), time_cell_ids)
np.save(os.path.join(save_dir, f'ramping_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), ramping_cell_ids)






