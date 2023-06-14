import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis.utils_analysis import sort_resp, plot_sorted_averaged_resp, single_cell_visualization, time_decode_lin_reg
from analysis.utils_time_ramp import lin_reg_ramping, skaggs_temporal_information, trial_reliability_vs_shuffle_score
from analysis.utils_mutual_info import joint_encoding_information_time_stimulus
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")


def plot_time_decoding_across_seeds(t_test_dict, t_test_pred_dict, len_delay):
    print("Plot time decoding across seeds...")
    t_test_arr = np.array(list(t_test_dict.values())).flatten()
    t_test_pred_arr = np.array(list(t_test_pred_dict.values())).flatten()
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
    ax.set_title(f'Time decoding pooled across {len(t_test_dict)} seeds')
    ax.set_xlim((np.min(t_test_pred_arr),41))
    ax.set_ylim((np.min(t_test_pred_arr),np.max(t_test_pred_arr)))
    fig.savefig(os.path.join(save_dir, f'time_decoding_pooled_{len(t_test_dict)}seeds_{n_total_episodes}_{n_shuffle}_{percentile}.svg'), dpi=300)


def plot_mutual_information_across_seeds(info_dict):
    print("Plot mutual information across seeds...")
    info_arr = np.vstack(info_dict.values())
    n_total_neurons = np.shape(info_arr)[0]
    breakpoint()
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
    plt.savefig(os.path.join(save_dir, f'joint_encoding_info_pooled_{len(info_dict)}seeds_{n_total_episodes}_{n_shuffle}_{percentile}.svg'))


data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl1d_og/mem_40_lstm_128_0.0001'
seed_list = []
for file in os.listdir(data_dir):
    if re.match(f'mem_40_lstm_128_0.0001_seed_\d+_epi199999.pt_data.npz', file):
        seed_list.append(int(file.split('_')[6]))
seed_list = sorted(seed_list)
n_total_episodes = 5000
len_delay = 40
n_neurons = 128
n_shuffle = 10
percentile = 99

seed_list = seed_list[:2]

save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_2/tunl1d'
os.makedirs(save_dir, exist_ok=True)
load = True

if load:
    t_test_dict = np.load(os.path.join(save_dir, f't_test_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), allow_pickle=True).item()
    t_test_pred_dict = np.load(os.path.join(save_dir, f't_test_pred_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), allow_pickle=True).item()
    info_dict = np.load(os.path.join(save_dir, f'info_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), allow_pickle=True).item()
else:
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
        data = np.load(os.path.join(data_dir, f'mem_40_lstm_128_0.0001_seed_{each_seed}_epi199999.pt_data.npz'), allow_pickle=True)
        stim = data['stim']
        first_action = data['first_action']
        delay_resp = data['delay_resp']
        left_stim_resp = delay_resp[stim == 0]
        right_stim_resp = delay_resp[stim == 1]
        # Normalize the delay response based on the maximum response of each neuron
        reshape_resp = np.reshape(delay_resp, (n_total_episodes*len_delay, n_neurons))
        reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
        delay_resp = np.reshape(reshape_resp, (n_total_episodes, len_delay, n_neurons))

        # Sort and plot the response
        print("Sort and plot the response...")
        cell_nums, sorted_resp = sort_resp(delay_resp, norm=True)
        plot_sorted_averaged_resp(cell_nums, sorted_resp, title='tiling_resp', remove_nan=True, save_dir=seed_save_dir, save=True)

        # visualize single unit response
        print("visualize single unit response...")
        single_cell_visualization(delay_resp, stim, cell_nums, type='all', save_dir=seed_save_dir)

        time_cell_ids[each_seed] = {}
        ramping_cell_ids[each_seed] = {}

        # Identify reliable cells
        print("Identify reliable cells...")
        trial_reliability_score_result_l, trial_reliability_score_threshold_result_l = trial_reliability_vs_shuffle_score(left_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
        trial_reliable_cell_bool_l = trial_reliability_score_result_l >= trial_reliability_score_threshold_result_l
        trial_reliable_cell_num_l = np.where(trial_reliable_cell_bool_l)[0]
        trial_reliability_score_result_r, trial_reliability_score_threshold_result_r = trial_reliability_vs_shuffle_score(right_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
        trial_reliable_cell_bool_r = trial_reliability_score_result_r >= trial_reliability_score_threshold_result_r
        trial_reliable_cell_num_r = np.where(trial_reliable_cell_bool_r)[0]

        # Identify ramping cells
        print("Identify ramping cells...")
        p_result_l, slope_result_l, intercept_result_l, R_result_l = lin_reg_ramping(left_stim_resp, plot=True, save_dir=save_dir, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_left')
        ramp_cell_bool_l = np.logical_and(p_result_l<=0.05, np.abs(R_result_l)>=0.9)
        cell_nums_ramp_l = np.where(ramp_cell_bool_l)[0]
        p_result_r, slope_result_r, intercept_result_r, R_result_r = lin_reg_ramping(right_stim_resp, plot=True, save_dir=save_dir, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_right')
        ramp_cell_bool_r = np.logical_and(p_result_r<=0.05, np.abs(R_result_r)>=0.9)
        cell_nums_ramp_r = np.where(ramp_cell_bool_r)[0]
        ramping_cell_ids[each_seed]['left'] = np.intersect1d(cell_nums_ramp_l, trial_reliable_cell_num_l)
        ramping_cell_ids[each_seed]['right'] = np.intersect1d(cell_nums_ramp_r, trial_reliable_cell_num_r)
        # ramping cell is either left or right ramping cells
        ramping_cell_ids[each_seed]['total'] = np.union1d(ramping_cell_ids[each_seed]['left'], ramping_cell_ids[each_seed]['right'])

        # Identify time cells
        print("Identify time cells...")
        I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, ramp_cell_bool_l, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_left', save_dir=save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)  # takes 8 hours to do 1000 shuffles for 256 neurons
        high_temporal_info_cell_bool_l = I_result_l > I_threshold_result_l
        high_temporal_info_cell_nums_l = np.where(high_temporal_info_cell_bool_l)[0]
        I_result_r, I_threshold_result_r = skaggs_temporal_information(right_stim_resp, ramp_cell_bool_r, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_right', save_dir=save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)
        high_temporal_info_cell_bool_r = I_result_r > I_threshold_result_r
        high_temporal_info_cell_nums_r = np.where(high_temporal_info_cell_bool_r)[0]
        time_cell_ids[each_seed]['left'] = np.intersect1d(high_temporal_info_cell_nums_l, trial_reliable_cell_num_l)
        time_cell_ids[each_seed]['right'] = np.intersect1d(high_temporal_info_cell_nums_r, trial_reliable_cell_num_r)
        # time cell is either left or right time cells
        time_cell_ids[each_seed]['total'] = np.union1d(time_cell_ids[each_seed]['left'], time_cell_ids[each_seed]['right'])

        # Decode time
        print("Decode time...")
        t_test_dict[each_seed], t_test_pred_dict[each_seed] = time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title='all_cells', save_dir=seed_save_dir, save=True)

        # Mutual Information
        print("Mutual Information...")
        info_dict[each_seed] = joint_encoding_information_time_stimulus(delay_resp, stim, save_dir=seed_save_dir,title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}', logInfo=False, save=True)

        # for each 10 seeds, plot the results
        if (i_seed+1) % 10 == 0:
            # plot time decoding
            plot_time_decoding_across_seeds(t_test_dict, t_test_pred_dict, len_delay)
            # plot mutual information
            plot_mutual_information_across_seeds(info_dict)
    # Save results
    print("Save results..")
    np.save(os.path.join(save_dir, f't_test_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_dict)
    np.save(os.path.join(save_dir, f't_test_pred_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), t_test_pred_dict)
    np.save(os.path.join(save_dir, f'info_dict_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), info_dict)
    np.save(os.path.join(save_dir, f'time_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), time_cell_ids)
    np.save(os.path.join(save_dir, f'ramping_cell_ids_{n_total_episodes}_{n_shuffle}_{percentile}.npy'), ramping_cell_ids)



