import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from scipy.stats import kruskal
from analysis.utils_analysis import sort_resp, plot_sorted_averaged_resp, single_cell_visualization, \
    time_decode_lin_reg, plot_decode_sample_from_single_time, decode_sample_from_single_time, plot_sorted_in_same_order, \
    identify_splitter_cells_ANOVA, identify_splitter_cells_discriminability
from analysis.utils_time_ramp import lin_reg_ramping, skaggs_temporal_information, trial_reliability_vs_shuffle_score, \
    plot_r_tuning_curves
from analysis.utils_mutual_info import joint_encoding_information_time_stimulus, construct_ratemap, \
    calculate_mutual_information, calculate_shuffled_mutual_information, plot_mutual_info_distribution, \
    joint_encoding_info, plot_joint_encoding_information, plot_stimulus_selective_place_cells, \
    decode_sample_from_trajectory, convert_loc_to_idx, joint_encoding_information_time_stimulus_location
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

# argparse each seed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--untrained', type=bool, default=False, help='whether to use untrained model')
args = parser.parse_args()
each_seed = args.seed
untrained = args.untrained

data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl2d/mem_40_lstm_256_5e-06'
n_total_episodes = 5000
len_delay = 40
n_neurons = 256
n_shuffle = 100
percentile = 99

#mem_40_lstm_256_5e-06_seed_153_epi79999.pt_data.npz
#seed_110_untrained_agent_data.npz

if untrained:
    save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_7/tunl2d_untrained_weight_frozen'
else:
    save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_7/tunl2d'
os.makedirs(save_dir, exist_ok=True)

print(f'====================== Analyzing seed {each_seed} ...======================================')
seed_save_dir = os.path.join(save_dir, f'seed_{each_seed}')
if not os.path.exists(seed_save_dir):
    os.makedirs(seed_save_dir, exist_ok=True)
# load the data
print("load the data...")
if untrained:
    data = np.load(os.path.join(data_dir, f'seed_{each_seed}_untrained_agent_weight_frozen_data.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(data_dir, f'mem_40_lstm_256_5e-06_seed_{each_seed}_epi79999.pt_data.npz'), allow_pickle=True)
# ['stim', 'choice', 'ct', 'delay_loc', 'delay_resp_hx', 'delay_resp_cx', 'epi_nav_reward', 'ideal_nav_rwds']
stim = data['stim']
choice = data['choice']
ct = data['ct']
delay_loc = data['delay_loc']
delay_resp_hx = data['delay_resp_hx']
delay_resp_cx = data['delay_resp_cx']
epi_nav_reward = data['epi_nav_reward']
ideal_nav_rwds = data['ideal_nav_rwds']
delay_resp = delay_resp_hx
# Normalize the delay response based on the maximum response of each neuron
reshape_resp = np.reshape(delay_resp, (n_total_episodes*len_delay, n_neurons))
# remove neurons with ptp==0
ptp = np.ptp(reshape_resp, axis=0, keepdims=True)
zero_ptp_neurons = np.unique(np.where(ptp == 0)[1])
reshape_resp = np.delete(reshape_resp, zero_ptp_neurons, axis=1)
n_neurons = np.shape(reshape_resp)[1]
print(f'Zero ptp neurons: {zero_ptp_neurons}. Number of neurons after removing zero ptp neurons: {n_neurons}')

# normalize the response
reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
delay_resp = np.reshape(reshape_resp, (n_total_episodes, len_delay, n_neurons))
left_stim_resp = delay_resp[np.all(stim == [1, 1], axis=1)]
right_stim_resp = delay_resp[np.any(stim != [1, 1], axis=1)]
left_stim_loc = delay_loc[np.all(stim == [1, 1], axis=1)]  # delay locations on stim==left trials
right_stim_loc = delay_loc[np.any(stim != [1, 1], axis=1)]
binary_stim = np.ones(np.shape(stim)[0])
binary_stim[np.all(stim == [1, 1], axis=1)] = 0  # 0 is L, 1 is right

ratemap_left_sti, spatial_occupancy_left_sti = construct_ratemap(left_stim_resp, left_stim_loc)
ratemap_right_sti, spatial_occupancy_right_sti = construct_ratemap(right_stim_resp, right_stim_loc)
mutual_info_left_sti = calculate_mutual_information(ratemap_left_sti, spatial_occupancy_left_sti)
mutual_info_right_sti = calculate_mutual_information(ratemap_right_sti, spatial_occupancy_right_sti)
delay_loc_idx = convert_loc_to_idx(delay_loc)

print("Plot splitter cells... SAVE AUTOMATICALLY")
plot_stimulus_selective_place_cells(mutual_info_left_sti, ratemap_left_sti, mutual_info_right_sti, ratemap_right_sti, save_dir=seed_save_dir, normalize_ratemaps=True)

print("identify splitter cells with ANOVA...")
splitter_cell_ids_ANOVA, anova_results = identify_splitter_cells_ANOVA(delay_resp, delay_loc_idx, binary_stim)
print("splitter cells with ANOVA: ", splitter_cell_ids_ANOVA)
np.save(os.path.join(seed_save_dir, 'splitter_cell_ids_ANOVA.npy'), splitter_cell_ids_ANOVA)
np.save(os.path.join(seed_save_dir, 'anova_results.npy'), anova_results)

print("identify splitter cells with discriminability...")
splitter_cell_ids_dis, discriminability_results = identify_splitter_cells_discriminability(ratemap_left_sti, ratemap_right_sti, n_shuffles=100, percentile=95)
print("splitter cells with discriminability: ", splitter_cell_ids_dis)
np.save(os.path.join(seed_save_dir, 'splitter_cell_ids_dis.npy'), splitter_cell_ids_dis)
np.save(os.path.join(seed_save_dir, 'discriminability_results.npy'), discriminability_results)

# SxTxL Mutual information analysis
print("SxTxL Mutual information analysis...")
stl_info = joint_encoding_information_time_stimulus_location(delay_resp, delay_loc_idx, binary_stim, save_dir=seed_save_dir, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}', save=True)
np.save(os.path.join(seed_save_dir, 'stl_mutual_info.npy'), stl_info)

print("r anaylsis...")
r_arr = plot_r_tuning_curves(left_stim_resp, right_stim_resp, 'left', 'right', save_dir=seed_save_dir)
np.save(os.path.join(seed_save_dir, 'r_arr.npy'), r_arr=r_arr)

# decode stimulus
print("decode stimulus...")
accuracies, accuracies_shuff = decode_sample_from_single_time(delay_resp, binary_stim, n_fold=5)
np.save(os.path.join(seed_save_dir, 'accuracies.npy'), accuracies)
np.save(os.path.join(seed_save_dir, 'accuracies_shuff.npy'), accuracies_shuff)

# Sort and plot the response
print("Sort and plot the response...")
cell_nums, sorted_resp = sort_resp(delay_resp, norm=True)
plot_sorted_averaged_resp(cell_nums, sorted_resp, title='tiling_resp', remove_nan=True, save_dir=seed_save_dir, save=True)
plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title="all_cells", len_delay=len_delay, n_neurons=n_neurons, save_dir=seed_save_dir, save=True)

# visualize single unit response
print("visualize single unit response...")
single_cell_visualization(delay_resp, binary_stim, cell_nums, type='all', save_dir=seed_save_dir)  # temporal tuning

# Decode time
print("Decode time...")
t_test_seed, t_test_pred_seed = time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title='all_cells', save_dir=seed_save_dir, save=True)
np.save(os.path.join(seed_save_dir, 't_test_seed.npy'), t_test_seed)
np.save(os.path.join(seed_save_dir, 't_test_pred_seed.npy'), t_test_pred_seed)

ramping_cell_ids_seed = {}
time_cell_ids_seed = {}
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
p_result_l, slope_result_l, intercept_result_l, R_result_l = lin_reg_ramping(left_stim_resp, plot=True, save_dir=seed_save_dir, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_left')
ramp_cell_bool_l = np.logical_and(p_result_l<=0.05, np.abs(R_result_l)>=0.9)
cell_nums_ramp_l = np.where(ramp_cell_bool_l)[0]
p_result_r, slope_result_r, intercept_result_r, R_result_r = lin_reg_ramping(right_stim_resp, plot=True, save_dir=seed_save_dir, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_right')
ramp_cell_bool_r = np.logical_and(p_result_r<=0.05, np.abs(R_result_r)>=0.9)
cell_nums_ramp_r = np.where(ramp_cell_bool_r)[0]
ramping_cell_ids_seed['left'] = np.intersect1d(cell_nums_ramp_l, trial_reliable_cell_num_l)
ramping_cell_ids_seed['right'] = np.intersect1d(cell_nums_ramp_r, trial_reliable_cell_num_r)
# ramping cell is either left or right ramping cells
ramping_cell_ids_seed['total'] = np.union1d(ramping_cell_ids_seed['left'], ramping_cell_ids_seed['right'])

# Identify time cells
print("Identify time cells...")
I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, ramp_cell_bool_l, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_left', save_dir=seed_save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)  # takes 8 hours to do 1000 shuffles for 256 neurons
high_temporal_info_cell_bool_l = I_result_l > I_threshold_result_l
high_temporal_info_cell_nums_l = np.where(high_temporal_info_cell_bool_l)[0]
I_result_r, I_threshold_result_r = skaggs_temporal_information(right_stim_resp, ramp_cell_bool_r, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_right', save_dir=seed_save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)
high_temporal_info_cell_bool_r = I_result_r > I_threshold_result_r
high_temporal_info_cell_nums_r = np.where(high_temporal_info_cell_bool_r)[0]
time_cell_ids_seed['left'] = np.intersect1d(high_temporal_info_cell_nums_l, trial_reliable_cell_num_l)
time_cell_ids_seed['right'] = np.intersect1d(high_temporal_info_cell_nums_r, trial_reliable_cell_num_r)
# time cell is either left or right time cells
time_cell_ids_seed['total'] = np.union1d(time_cell_ids_seed['left'], time_cell_ids_seed['right'])
np.save(os.path.join(seed_save_dir, 'time_cell_ids_seed.npy'), time_cell_ids_seed)
np.save(os.path.join(seed_save_dir, 'ramping_cell_ids_seed.npy'), ramping_cell_ids_seed)


# ========mutual information analysis=======
# Space x time mutual information
print("Space x time mutual information...")
ratemap, spatial_occupancy = construct_ratemap(delay_resp, delay_loc)
mutual_info = calculate_mutual_information(ratemap, spatial_occupancy)
shuffled_mutual_info = calculate_shuffled_mutual_information(delay_resp, delay_loc, n_total_episodes)

plot_mutual_info_distribution(mutual_info, title='all_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info, save_dir=seed_save_dir, save=True)

joint_encoding_info(delay_resp, delay_loc, save_dir=seed_save_dir, recalculate=True)  # saved joint_encoding.npz to save_dir
plot_joint_encoding_information(save_dir=seed_save_dir, title='all_cells', save=True)

# Mutual Information
print("Mutual Information...")
info_seed = joint_encoding_information_time_stimulus(delay_resp, binary_stim, save_dir=seed_save_dir,title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}', logInfo=False, save=True)
np.save(os.path.join(seed_save_dir, 'st_info_seed.npy'), info_seed)

print("decode sample from trajectory...")
trajectory_stim_decoding_accuracy = decode_sample_from_trajectory(delay_loc, stim, save_dir=seed_save_dir, save=True)
np.save(os.path.join(seed_save_dir, 'trajectory_stim_decoding_accuracy.npy'), trajectory_stim_decoding_accuracy)

print("analysis complete")

