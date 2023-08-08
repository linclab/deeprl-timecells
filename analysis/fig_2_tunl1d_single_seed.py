import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from scipy.stats import kruskal
from analysis.utils_analysis import sort_resp, plot_sorted_averaged_resp, single_cell_visualization, \
    time_decode_lin_reg, plot_decode_sample_from_single_time, decode_sample_from_single_time
from analysis.utils_time_ramp import lin_reg_ramping, skaggs_temporal_information, trial_reliability_vs_shuffle_score, \
    plot_r_tuning_curves
from analysis.utils_mutual_info import joint_encoding_information_time_stimulus
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

data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl1d_og/mem_40_lstm_128_0.0001'
n_total_episodes = 5000
len_delay = 40
n_neurons = 128
n_shuffle = 100
percentile = 99

if untrained:
    save_dir = '/network/scratch/l/lindongy/timecell/figures/nocue_dim4/tunl1d_untrained_weight_frozen'
else:
    save_dir = '/network/scratch/l/lindongy/timecell/figures/nocue_dim4/tunl1d'
os.makedirs(save_dir, exist_ok=True)

ramping_cell_ids = {}
time_cell_ids = {}
t_test_dict = {}
t_test_pred_dict = {}
info_dict = {}

print(f'====================== Analyzing seed {each_seed} ...======================================')
seed_save_dir = os.path.join(save_dir, f'seed_{each_seed}')
if not os.path.exists(seed_save_dir):
    os.makedirs(seed_save_dir, exist_ok=True)
# load the data
print("load the data...")
if untrained:
    data = np.load(os.path.join(data_dir, f'seed_{each_seed}_untrained_agent_weight_frozen_data.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(data_dir, f'mem_40_lstm_128_0.0001_seed_{each_seed}_epi199999.pt_data.npz'), allow_pickle=True)
    
stim = data['stim']
first_action = data['first_action']
delay_resp = data['delay_resp']

# Normalize the delay response based on the maximum response of each neuron
reshape_resp = np.reshape(delay_resp, (n_total_episodes*len_delay, n_neurons))
reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
delay_resp = np.reshape(reshape_resp, (n_total_episodes, len_delay, n_neurons))
left_stim_resp = delay_resp[stim == 0]
right_stim_resp = delay_resp[stim == 1]

print("r anaylsis...")
r_arr = plot_r_tuning_curves(left_stim_resp, right_stim_resp, 'left', 'right', save_dir=save_dir)
# save the r_arr
np.save(os.path.join(seed_save_dir, 'r_arr.npy'), r_arr)


# decode stimulus
print("decode stimulus...")
accuracies, accuracies_shuff = decode_sample_from_single_time(delay_resp, stim, n_fold=5)
np.save(os.path.join(seed_save_dir, 'accuracies.npy'), accuracies)
np.save(os.path.join(seed_save_dir, 'accuracies_shuff.npy'), accuracies_shuff)



# Sort and plot the response
print("Sort and plot the response...")
cell_nums, sorted_resp = sort_resp(delay_resp, norm=True)
plot_sorted_averaged_resp(cell_nums, sorted_resp, title='tiling_resp', remove_nan=True, save_dir=seed_save_dir, save=True)

# visualize single unit response
print("visualize single unit response...")
single_cell_visualization(delay_resp, stim, cell_nums, type='all', save_dir=seed_save_dir)

# Decode time
print("Decode time...")
t_test_seed, t_test_pred_seed = time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title='all_cells', save_dir=seed_save_dir, save=True)

# Mutual Information
print("Mutual Information...")
info_seed = joint_encoding_information_time_stimulus(delay_resp, stim, save_dir=seed_save_dir,title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}', logInfo=False, save=True)
np.save(os.path.join(seed_save_dir, 'info_seed.npy'), info_seed)

time_cell_ids_seed = {}
ramping_cell_ids_seed = {}

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
ramping_cell_ids_seed['left'] = np.intersect1d(cell_nums_ramp_l, trial_reliable_cell_num_l)
ramping_cell_ids_seed['right'] = np.intersect1d(cell_nums_ramp_r, trial_reliable_cell_num_r)
# ramping cell is either left or right ramping cells
ramping_cell_ids_seed['total'] = np.union1d(ramping_cell_ids_seed['left'], ramping_cell_ids_seed['right'])

# Identify time cells
print("Identify time cells...")
I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, ramp_cell_bool_l, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_left', save_dir=save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)  # takes 8 hours to do 1000 shuffles for 256 neurons
high_temporal_info_cell_bool_l = I_result_l > I_threshold_result_l
high_temporal_info_cell_nums_l = np.where(high_temporal_info_cell_bool_l)[0]
I_result_r, I_threshold_result_r = skaggs_temporal_information(right_stim_resp, ramp_cell_bool_r, title=f'{each_seed}_{n_total_episodes}_{n_shuffle}_{percentile}_right', save_dir=save_dir, n_shuff=n_shuffle, percentile=percentile, plot=True)
high_temporal_info_cell_bool_r = I_result_r > I_threshold_result_r
high_temporal_info_cell_nums_r = np.where(high_temporal_info_cell_bool_r)[0]
time_cell_ids_seed['left'] = np.intersect1d(high_temporal_info_cell_nums_l, trial_reliable_cell_num_l)
time_cell_ids_seed['right'] = np.intersect1d(high_temporal_info_cell_nums_r, trial_reliable_cell_num_r)
# time cell is either left or right time cells
time_cell_ids_seed['total'] = np.union1d(time_cell_ids_seed['left'], time_cell_ids_seed['right'])

# save the results
print("Save the results...")
np.save(os.path.join(seed_save_dir, 't_test_seed.npy'), t_test_seed)
np.save(os.path.join(seed_save_dir, 't_test_pred_seed.npy'), t_test_pred_seed)
# np.save(os.path.join(seed_save_dir, 'info_seed.npy'), info_seed)
np.save(os.path.join(seed_save_dir, 'time_cell_ids_seed.npy'), time_cell_ids_seed)
np.save(os.path.join(seed_save_dir, 'ramping_cell_ids_seed.npy'), ramping_cell_ids_seed)
print("analysis complete")


        