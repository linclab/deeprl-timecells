from analysis_helper import *
from world import *
from linclab_utils import plot_utils

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

print('Analysis finished')
