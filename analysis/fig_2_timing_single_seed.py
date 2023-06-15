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


# argparse each seed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
each_seed = args.seed


data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/timing/lstm_128_1e-05'
save_dir = '/network/scratch/l/lindongy/timecell/figures/fig_2/timing1d'
n_neurons = 128
n_shuffle = 100
percentile = 99

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

print("mutual information analysis...")
info_dict_seed = {}
# Mutual information
stim_set = np.sort(np.unique(stim))
for stim_len in stim_set:
    binary_stim = np.concatenate((np.zeros(np.sum(stim[:, 0] == stim_len)), np.ones(np.sum(stim[:, 1] == stim_len))))
    resp_for_stim_len = np.concatenate((stim1_resp[stim[:, 0] == stim_len, :stim_len, :], stim2_resp[stim[:, 1] == stim_len, :stim_len, :]), axis=0)
    info_dict_seed[stim_len] = joint_encoding_information_time_stimulus(resp_for_stim_len, binary_stim, seed_save_dir,title=f'stim{stim_len}', logInfo=False, save=True)

t_test_dict_seed = {}
t_test_pred_dict_seed = {}
for (resp, stimulus, label) in zip([stim1_resp,stim2_resp, delay_resp], [stim[:,0],stim[:,1], None], ['stimulus_1', 'stimulus_2', 'delay']):
    # Decode time
    print(f"time decoding for {label}...")
    len_delay = 40 if label != 'delay' else 20
    t_test_dict_seed[label], t_test_pred_dict_seed[label] = time_decode_lin_reg(resp, len_delay, n_neurons, 1000, title=label+'_all_cells', save_dir=seed_save_dir, save=True)

    # Sort and plot the response
    print(f"Sort and plot the response for {label}...")
    cell_nums, sorted_resp = sort_resp(resp, norm=True)
    plot_sorted_averaged_resp(cell_nums, sorted_resp, title=label+'_tiling_resp', remove_nan=True, save_dir=seed_save_dir, save=True)

# Visualize single cell response
print("visualizing single cell response...")
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


time_cell_ids_seed = {}
ramping_cell_ids_seed = {}
for (resp, stimulus, label) in zip([stim1_resp,stim2_resp], [stim[:,0],stim[:,1]], ['stimulus_1', 'stimulus_2']):
    # Identify trial reliable cells
    trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_vs_shuffle_score_varying_duration(resp, stimulus, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
    trial_reliable_cell_bool = trial_reliability_score_result >= trial_reliability_score_threshold_result
    trial_reliable_cell_num = np.where(trial_reliable_cell_bool)[0]
    # Identify ramping cells
    p_result, slope_result, intercept_result, R_result = lin_reg_ramping_varying_duration(resp, plot=True, save_dir=seed_save_dir, title=f'{n_shuffle}_{percentile}_{label}')
    ramp_cell_bool = np.logical_and(p_result<=0.05, np.abs(R_result)>=0.9)
    cell_nums_ramp = np.where(ramp_cell_bool)[0]
    ramping_cell_ids_seed[label] = np.intersect1d(trial_reliable_cell_num, cell_nums_ramp)
    # Identify time cells
    RB_result, z_RB_threshold_result = ridge_to_background_varying_duration(resp, stimulus,  ramp_cell_bool, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=seed_save_dir, title=f'{n_shuffle}_{percentile}_{label}')
    time_cell_bool = RB_result > z_RB_threshold_result
    cell_nums_time = np.where(time_cell_bool)[0]
    time_cell_ids_seed[label] = np.intersect1d(trial_reliable_cell_num, cell_nums_time)

time_cell_ids_seed['total'] = np.union1d(time_cell_ids_seed['stimulus_1'], time_cell_ids_seed['stimulus_2'])
ramping_cell_ids_seed['total'] = np.union1d(ramping_cell_ids_seed['stimulus_1'], ramping_cell_ids_seed['stimulus_2'])



# save all the dicts as npy
np.save(os.path.join(seed_save_dir, 'info_dict_seed.npy'), info_dict_seed)
np.save(os.path.join(seed_save_dir, 't_test_dict_seed.npy'), t_test_dict_seed)
np.save(os.path.join(seed_save_dir, 't_test_pred_dict_seed.npy'), t_test_pred_dict_seed)
np.save(os.path.join(seed_save_dir, 'time_cell_ids_seed.npy'), time_cell_ids_seed)
np.save(os.path.join(seed_save_dir, 'ramping_cell_ids_seed.npy'), ramping_cell_ids_seed)

