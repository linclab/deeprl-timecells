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
n_shuffle = 3
percentile = 99

seed_list = seed_list[:2]

save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_2/timing1d'
os.makedirs(save_dir, exist_ok=True)
load = False

ramping_cell_ids = {}
time_cell_ids = {}
t_test_dict = {}
t_test_pred_dict = {}
info_dict = {}
for i_seed, each_seed in enumerate(seed_list):
    print(f'====================== Analyzing seed {each_seed} ...======================================')
    seed_save_dir = os.path.join(save_dir, f'seed_{each_seed}')
    if not os.path.exists(seed_save_dir):
        os.makedirs(seed_save_dir)
    # load the data
    print("load the data...")
    data = np.load(os.path.join(data_dir, f'lstm_128_1e-05_seed_{each_seed}_epi149999.pt_data.npz'), allow_pickle=True)
    action_hist = data["action_hist"]
    correct_trials = data["correct_trial"]
    stim = data["stim"]
    stim1_resp = data["stim1_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    stim2_resp = data["stim2_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    delay_resp = data["delay_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    n_total_episodes = np.shape(stim)[0]

    # Visualize single cell response
    single_cell_temporal_tuning(stim, stim1_resp, stim2_resp, save_dir=seed_save_dir)
    
    # Normalize the delay response based on the maximum response of each neuron
    for i_neuron in range(n_neurons):
        # swap 0's in stim1_resp and stim2_resp with nan
        stim1_resp[:, :, i_neuron][stim1_resp[:, :, i_neuron] == 0] = np.nan
        stim2_resp[:, :, i_neuron][stim2_resp[:, :, i_neuron] == 0] = np.nan
        # normalize across stim1_resp, stim2_resp, and delay_resp
        min_act = np.nanmin(np.concatenate((stim1_resp[:, :, i_neuron], stim2_resp[:, :, i_neuron], delay_resp[:, :, i_neuron]), axis=1))
        max_act = np.nanmax(np.concatenate((stim1_resp[:, :, i_neuron], stim2_resp[:, :, i_neuron], delay_resp[:, :, i_neuron]), axis=1))
        stim1_resp[:, :, i_neuron] = (stim1_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        stim2_resp[:, :, i_neuron] = (stim2_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        delay_resp[:, :, i_neuron] = (delay_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        # swap nan's back to 0's
        stim1_resp[:, :, i_neuron][np.isnan(stim1_resp[:, :, i_neuron])] = 0
        stim2_resp[:, :, i_neuron][np.isnan(stim2_resp[:, :, i_neuron])] = 0
        delay_resp[:, :, i_neuron][np.isnan(delay_resp[:, :, i_neuron])] = 0

    time_cell_ids[each_seed] = {}
    ramping_cell_ids[each_seed] = {}
    for (resp, stimulus, label) in zip([stim1_resp,stim2_resp], [stim[:,0],stim[:,1]], ['stimulus_1', 'stimulus_2']):
        # Identify trial reliable cells
        trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_vs_shuffle_score_varying_duration(resp, stimulus, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
        trial_reliable_cell_bool = trial_reliability_score_result >= trial_reliability_score_threshold_result
        trial_reliable_cell_num = np.where(trial_reliable_cell_bool)[0]
        # Identify ramping cells
        p_result, slope_result, intercept_result, R_result = lin_reg_ramping_varying_duration(resp, plot=True, save_dir=seed_save_dir, title=f'{n_shuffle}_{percentile}_{label}')
        ramp_cell_bool = np.logical_and(p_result<=0.05, np.abs(R_result)>=0.9)
        cell_nums_ramp = np.where(ramp_cell_bool)[0]
        ramping_cell_ids[each_seed][label] = np.intersect1d(trial_reliable_cell_num, cell_nums_ramp)
        # Identify time cells
        RB_result, z_RB_threshold_result = ridge_to_background_varying_duration(resp, stimulus,  ramp_cell_bool, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=seed_save_dir, title=f'{n_shuffle}_{percentile}_{label}')
        time_cell_bool = RB_result > z_RB_threshold_result
        cell_nums_time = np.where(time_cell_bool)[0]
        time_cell_ids[each_seed][label] = np.intersect1d(trial_reliable_cell_num, cell_nums_time)

    time_cell_ids[each_seed]['total'] = np.union1d(time_cell_ids[each_seed]['stimulus_1'], time_cell_ids[each_seed]['stimulus_2'])
    ramping_cell_ids[each_seed]['total'] = np.union1d(ramping_cell_ids[each_seed]['stimulus_1'], ramping_cell_ids[each_seed]['stimulus_2'])
    
    info_dict[each_seed] = {}
    # Mutual information
    stim_set = np.sort(np.unique(stim))
    for stim_len in stim_set:
        binary_stim = np.concatenate((np.zeros(np.sum(stim[:, 0] == stim_len)), np.ones(np.sum(stim[:, 1] == stim_len))))
        resp_for_stim_len = np.concatenate((stim1_resp[stim[:, 0] == stim_len, :stim_len, :], stim2_resp[stim[:, 1] == stim_len, :stim_len, :]), axis=0)
        info_dict[each_seed][stim_len] = joint_encoding_information_time_stimulus(resp_for_stim_len, binary_stim, seed_save_dir,title=f'stim{stim_len}', logInfo=False, save=True)

    t_test_dict[each_seed] = {}
    t_test_pred_dict[each_seed] = {}
    for (resp, stimulus, label) in zip([stim1_resp,stim2_resp, delay_resp], [stim[:,0],stim[:,1], None], ['stimulus_1', 'stimulus_2', 'delay']):
        # Decode time
        len_delay = 40 if label != 'delay' else 20
        t_test_dict[each_seed][label], t_test_pred_dict[each_seed][label] = time_decode_lin_reg(resp, len_delay, n_neurons, 1000, title=label+'_all_cells', save_dir=seed_save_dir, save=True)

        # Sort and plot the response
        print("Sort and plot the response...")
        cell_nums, sorted_resp = sort_resp(resp, norm=True)
        plot_sorted_averaged_resp(cell_nums, sorted_resp, title=label+'_tiling_resp', remove_nan=True, save_dir=seed_save_dir, save=True)

        if (i_seed+1) % 2 == 0:
            plot_time_decoding_across_seeds(t_test_dict, t_test_pred_dict, label)


    if (i_seed+1) % 2 == 0:
        # plot mutual information
        plot_mutual_information_across_seeds(info_dict)

        # plot counts of time and ramping cells
        plot_count_time_and_ramping_cells(time_cell_ids, ramping_cell_ids)

# Save results
print("Save results..")
np.save(os.path.join(save_dir, f't_test_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_dict)
np.save(os.path.join(save_dir, f't_test_pred_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_pred_dict)
np.save(os.path.join(save_dir, f'info_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), info_dict)
np.save(os.path.join(save_dir, f'time_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), time_cell_ids)
np.save(os.path.join(save_dir, f'ramping_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), ramping_cell_ids)






