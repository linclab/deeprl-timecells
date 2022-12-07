import numpy as np
from analysis.cell_identification.time_ramp import separate_ramp_and_seq
from plot_utils import *
from expts.envs.tunl_1d import *
from linclab_utils import plot_utils

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

data = np.load('')  # data.npz file
label = 'non-spatial Mem TUNL'
global figpath
figpath = ''  # path to save figures


'''
# For non-Variable Delay
binary_stim = data['stim']  # n_episode
delay_resp_hx = data['delay_resp']  # n_episode x len_delay x n_neurons
binary_nonmatch = data['binary_nonmatch']

len_delay = 40
n_total_neurons = np.shape(delay_resp_hx)[2]

# Select units with large enough variation in its activation
#big_var_neurons = []
#for i_neuron in range(n_total_neurons):
#    if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
#        big_var_neurons.append(i_neuron)

#delay_resp = delay_resp_hx[:, 1:, [x for x in range(n_total_neurons) if x in big_var_neurons]]
delay_resp = delay_resp_hx[:, 1:, :]
'''

# For Variable Delay
#binary_stim = data['stim']  # n_episode
delay_resp = data['delay_resp']  # n_episode x len_delay x n_neurons
#binary_nonmatch = data['binary_nonmatch']
ld = data['len_delay']
resp_dict, counts = separate_vd_resp(delay_resp, ld)

len_delay = 60
delay_resp_hx = resp_dict[len_delay]
binary_stim = data['stim'][ld==len_delay]
binary_nonmatch = data['binary_nonmatch'][ld==len_delay]
n_total_neurons = np.shape(delay_resp_hx)[2]

# Select units with large enough variation in its activation
big_var_neurons = []
for i_neuron in range(n_total_neurons):
    if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
        big_var_neurons.append(i_neuron)

delay_resp = delay_resp_hx[:, 1:, [x for x in range(n_total_neurons) if x in big_var_neurons]]
#delay_resp = delay_resp_hx[:, 1:, :]
#plot_sorted_vd(resp_dict)



n_episodes = np.shape(delay_resp)[0]
len_delay = np.shape(delay_resp)[1]
n_neurons = np.shape(delay_resp)[2]
print(np.shape(delay_resp))
# separate left and right trials
left_stim_resp = delay_resp[binary_stim == 0]
right_stim_resp = delay_resp[binary_stim == 1]

#left_choice_resp = delay_resp[binary_choice == 0]
#right_choice_resp = delay_resp[binary_choice == 1]

#binary_nonmatch = np.ones(n_episodes)
#binary_nonmatch[binary_stim == binary_choice] = 0  # 0 is match, 1 is nonmatch

correct_resp = delay_resp[binary_nonmatch == 1]
incorrect_resp = delay_resp[binary_nonmatch == 0]

cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
    delay_resp, norm=True)
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_seq = delay_resp[:, :, cell_nums_seq]
left_stim_resp_ramp = delay_resp_ramp[binary_stim == 0]
right_stim_resp_ramp = delay_resp_ramp[binary_stim == 1]
left_stim_resp_seq = delay_resp_seq[binary_stim == 0]
right_stim_resp_seq = delay_resp_seq[binary_stim == 1]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)

print('Make a pie chart of neuron counts...')
#make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, n_total_neurons, figpath, label)


print('Sort avg resp analysis...')
#plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=label+' Sequence cells', remove_nan=True)
#plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=label+' Ramping cells', remove_nan=True)
#plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=label+' All cells', remove_nan=True)

# print('dim v delay t analysis...')
# plot_dim_vs_delay_t(left_stim_resp, title=label+' All cells (Left trials)', n_trials=10, var_explained=0.99)
# plot_dim_vs_delay_t(left_stim_resp_ramp, title=label+' Ramping cells (Left trials)', n_trials=10, var_explained=0.99)
# plot_dim_vs_delay_t(left_stim_resp_seq, title=label+' Sequence cells (Left trials)', n_trials=10,  var_explained=0.99)
# plot_dim_vs_delay_t(right_stim_resp, title=label+' All cells (Right trials)', n_trials=10, var_explained=0.99)
# plot_dim_vs_delay_t(right_stim_resp_ramp, title=label+' Ramping cells (Right trials)', n_trials=10, var_explained=0.99)
# plot_dim_vs_delay_t(right_stim_resp_seq, title=label+' Sequence cells (Right trials)', n_trials=10,  var_explained=0.99)

print('sort in same order analysis...')
#plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=label+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons)
#plot_sorted_in_same_order(left_stim_resp_seq, right_stim_resp_seq, 'Left', 'Right', big_title=label+' Sequence cells', len_delay=len_delay, n_neurons=n_seq_neurons)
#plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=label+' All cells', len_delay=len_delay, n_neurons=n_neurons)
#plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=label+' All cells ic', len_delay=len_delay, n_neurons=n_neurons)

print('decode stim analysis...')
plot_decode_sample_from_single_time(delay_resp, binary_stim, label + ' All Cells', n_fold=5, max_iter=100)
plot_decode_sample_from_single_time(delay_resp_ramp, binary_stim, label + ' Ramping Cells', n_fold=5, max_iter=100)
plot_decode_sample_from_single_time(delay_resp_seq, binary_stim, label + ' Sequence Cells', n_fold=7, max_iter=100)

print('decode time analysis...')
#time_decode(delay_resp, len_delay, n_neurons, 1000, title=label+' All cells', plot=True)
#time_decode(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=label+' Ramping cells', plot=True)
#time_decode(delay_resp_seq, len_delay, n_seq_neurons, 1000, title=label+' Sequence cells', plot=True)

# print('Single-cell visualization...')
# single_cell_visualization(delay_resp, binary_stim, cell_nums_ramp, type='ramp')
# single_cell_visualization(delay_resp, binary_stim, cell_nums_seq, type='seq')

# print('pca...')
# pca_analysis(delay_resp, binary_stim)

# print('single cell ratemaps...')
# plot_LvR_ratemaps(delay_resp, delay_loc, left_stim_resp, left_stim_loc, right_stim_resp, right_stim_loc, cell_nums_all,cell_nums_seq, cell_nums_ramp, label)

print('Analysis finished')