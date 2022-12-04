from expts.envs.tunl_2d import Tunl, Tunl_vd, Tunl_nomem, Tunl_nomem_vd
from linclab_utils import plot_utils
import numpy as np
from mutual_info.utils import *
from cell_identification.time_ramp import separate_ramp_and_seq
from plot_utils import make_piechart, plot_sorted_averaged_resp, plot_sorted_in_same_order, plot_dim_vs_delay_t, \
    single_cell_visualization
from analysis.decoder import plot_decode_sample_from_single_time, time_decode

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

data = np.load('')  # data.npz file
label = 'NoMem TUNL'


stim = data['stim']  # n_episode x 2
delay_resp_hx = data['delay_resp_hx']  # n_episode x len_delay x n_neurons
delay_resp_cx = data['delay_resp_cx']  # n_episode x len_delay x n_neurons
delay_loc = data['delay_loc']  # n_episode x len_delay x 2
choice = data['choice']  # n_episode x 2
action = data['action']  # n_episode x len_delay

# Select units with large enough variation in its activation
big_var_neurons = []
for i_neuron in range(512):
    if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
        big_var_neurons.append(i_neuron)

delay_resp = delay_resp_hx[:, 2:, [x for x in range(512) if x in big_var_neurons]]
delay_loc = delay_loc[:, 2:, :]

n_episodes = np.shape(delay_resp)[0]
len_delay = np.shape(delay_resp)[1]
n_neurons = np.shape(delay_resp)[2]

# separate left and right trials
left_stim_resp = delay_resp[np.all(stim == [1, 1], axis=1)]
right_stim_resp = delay_resp[np.any(stim != [1, 1], axis=1)]
left_stim_loc = delay_loc[np.all(stim == [1, 1], axis=1)]  # delay locations on stim==left trials
right_stim_loc = delay_loc[np.any(stim != [1, 1], axis=1)]

left_choice_resp = delay_resp[np.all(choice == [1, 1], axis=1)]
right_choice_resp = delay_resp[np.any(choice != [1, 1], axis=1)]
left_choice_loc = delay_loc[np.all(choice == [1, 1], axis=1)]  # delay locations on first_choice=left trials
right_choice_loc = delay_loc[np.any(choice != [1, 1], axis=1)]

binary_stim = np.ones(np.shape(stim)[0])
binary_stim[np.all(stim == [1, 1], axis=1)] = 0  # 0 is L, 1 is right

binary_choice = np.ones(np.shape(choice)[0])
binary_choice[np.all(choice == [1, 1], axis=1)] = 0  # 0 is L, 1 is right

binary_nonmatch = np.ones(np.shape(stim)[0])
binary_nonmatch[binary_stim == binary_choice] = 0  # 0 is match, 1 is nonmatch

correct_resp = delay_resp[binary_nonmatch == 1]
incorrect_resp = delay_resp[binary_nonmatch == 0]
correct_loc = delay_loc[binary_nonmatch == 1]  # delay locations on correct trials
incorrect_loc = delay_loc[binary_nonmatch == 0]

cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
    delay_resp, norm=True)
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_seq = delay_resp[:, :, cell_nums_seq]
left_stim_resp_ramp = delay_resp_ramp[np.all(stim == [1, 1], axis=1)]
right_stim_resp_ramp = delay_resp_ramp[np.any(stim != [1, 1], axis=1)]
left_stim_resp_seq = delay_resp_seq[np.all(stim == [1, 1], axis=1)]
right_stim_resp_seq = delay_resp_seq[np.any(stim != [1, 1], axis=1)]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)

print('Make a pie chart of neuron counts...')
make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, figpath, label)


print('Sort avg resp analysis...')
plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=label+' Sequence cells', remove_nan=True)
plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=label+' Ramping cells', remove_nan=True)
plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=label+' All cells', remove_nan=True)

print('dim v delay t analysis...')
plot_dim_vs_delay_t(left_stim_resp, title=label+' All cells (Left trials)', n_trials=10, var_explained=0.99)
plot_dim_vs_delay_t(left_stim_resp_ramp, title=label+' Ramping cells (Left trials)', n_trials=10, var_explained=0.99)
plot_dim_vs_delay_t(left_stim_resp_seq, title=label+' Sequence cells (Left trials)', n_trials=10,  var_explained=0.99)
plot_dim_vs_delay_t(right_stim_resp, title=label+' All cells (Right trials)', n_trials=10, var_explained=0.99)
plot_dim_vs_delay_t(right_stim_resp_ramp, title=label+' Ramping cells (Right trials)', n_trials=10, var_explained=0.99)
plot_dim_vs_delay_t(right_stim_resp_seq, title=label+' Sequence cells (Right trials)', n_trials=10,  var_explained=0.99)

print('sort in same order analysis...')
plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=label+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons)
plot_sorted_in_same_order(left_stim_resp_seq, right_stim_resp_seq, 'Left', 'Right', big_title=label+' Sequence cells', len_delay=len_delay, n_neurons=n_seq_neurons)
plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=label+' All cells', len_delay=len_delay, n_neurons=n_neurons)
plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=label+' All cells ic', len_delay=len_delay, n_neurons=n_neurons)

print('decode stim analysis...')
plot_decode_sample_from_single_time(delay_resp, binary_stim, label+' All Cells', n_fold=5, max_iter=100)
plot_decode_sample_from_single_time(delay_resp_ramp, binary_stim, label+' Ramping Cells', n_fold=5, max_iter=100)
plot_decode_sample_from_single_time(delay_resp_seq, binary_stim, label+' Sequence Cells', n_fold=7, max_iter=100)

print('decode time analysis...')
time_decode(delay_resp, len_delay, n_neurons, 1000, title=label+' All cells', plot=True)
time_decode(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=label+' Ramping cells', plot=True)
time_decode(delay_resp_seq, len_delay, n_seq_neurons, 1000, title=label+' Sequence cells', plot=True)

print('Single-cell visualization...')
single_cell_visualization(delay_resp, binary_stim, cell_nums_ramp, type='ramp')
single_cell_visualization(delay_resp, binary_stim, cell_nums_seq, type='seq')


# ==========================================

# Mutual information analysis
# ==========================================

# separate left and right trials
left_stim_resp = delay_resp[np.all(stim == [1, 1], axis=1)]
right_stim_resp = delay_resp[np.any(stim != [1, 1], axis=1)]
left_stim_loc = delay_loc[np.all(stim == [1, 1], axis=1)]  # delay locations on stim==left trials
right_stim_loc = delay_loc[np.any(stim != [1, 1], axis=1)]
# mutual information analysis
seq_cell_idx = separate_ramp_and_seq(delay_resp)[2]
ratemap, spatial_occupancy = construct_ratemap(delay_resp[:,:,seq_cell_idx], delay_loc)
mutual_info = calculate_mutual_information(ratemap, spatial_occupancy)
shuffled_mutual_info = calculate_shuffled_mutual_information(delay_resp, delay_loc, n_episodes)

plot_mutual_info_distribution(mutual_info, compare=True, shuffled_mutual_info=shuffled_mutual_info)

joint_encoding_info(delay_resp, delay_loc, analysis='selectivity', recalculate=True)
plot_joint_encoding_information()

ratemap_left_sti, spatial_occupancy_left_sti = construct_ratemap(left_stim_resp, left_stim_loc)
ratemap_right_sti, spatial_occupancy_right_sti = construct_ratemap(right_stim_resp, right_stim_loc)
mutual_info_left_sti = calculate_mutual_information(ratemap_left_sti, spatial_occupancy_left_sti)
mutual_info_right_sti = calculate_mutual_information(ratemap_right_sti, spatial_occupancy_right_sti)

plot_stimulus_selective_place_celss(mutual_info_left_sti, ratemap_left_sti, mutual_info_right_sti, ratemap_right_sti)

decode_sample_from_trajectory(delay_loc, stim)
print("Mutual info for ramping cells and sequence cells")
seq_cell_idx = separate_ramp_and_seq(delay_resp)[2]
ramp_cell_idx = separate_ramp_and_seq(delay_resp)[4]
nontime_cell_idx = [x for x in range(n_neurons) if (x not in seq_cell_idx and x not in ramp_cell_idx)]
print(np.shape(seq_cell_idx))
print(np.shape(ramp_cell_idx))
print(n_neurons)

delay_resp_seq = delay_resp[:,:,seq_cell_idx]
delay_resp_ramp = delay_resp[:,:,ramp_cell_idx]
delay_resp_nontime = delay_resp[:,:,nontime_cell_idx]

# ======= Starting Line 110 of analysis_place.py ==========
ratemap_seq, spatial_occupancy_seq = construct_ratemap(delay_resp_seq, delay_loc)
mutual_info_seq = calculate_mutual_information(ratemap_seq, spatial_occupancy_seq)
shuffled_mutual_info_seq = calculate_shuffled_mutual_information(delay_resp_seq, delay_loc, n_episodes)
#plot_mutual_info_distribution(mutual_info_seq, compare=True, shuffled_mutual_info=shuffled_mutual_info_seq)
print("seq: ", np.nanmean(mutual_info_seq))
print("shuffled seq: ", np.nanmean(shuffled_mutual_info_seq))

ratemap_ramp, spatial_occupancy_ramp = construct_ratemap(delay_resp_ramp, delay_loc)
mutual_info_ramp = calculate_mutual_information(ratemap_ramp, spatial_occupancy_ramp)
shuffled_mutual_info_ramp = calculate_shuffled_mutual_information(delay_resp_ramp, delay_loc, n_episodes)
#plot_mutual_info_distribution(mutual_info_ramp, compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp)
print("ramp: ", np.nanmean(mutual_info_ramp))
print("shuffled ramp: ", np.nanmean(shuffled_mutual_info_ramp))

ratemap_nontime, spatial_occupancy_nontime = construct_ratemap(delay_resp_nontime, delay_loc)
mutual_info_nontime = calculate_mutual_information(ratemap_nontime, spatial_occupancy_nontime)
shuffled_mutual_info_nontime = calculate_shuffled_mutual_information(delay_resp_nontime, delay_loc, n_episodes)
#plot_mutual_info_distribution(mutual_info_ramp, compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp)
print("ramp: ", np.nanmean(mutual_info_nontime))
print("shuffled ramp: ", np.nanmean(shuffled_mutual_info_nontime))


print('Analysis finished')
