from linclab_utils import plot_utils
from mutual_info.utils import *
from cell_identification.time_ramp import separate_ramp_and_seq
from analysis_utils import bin_rewards, make_piechart, plot_sorted_averaged_resp, plot_sorted_in_same_order, plot_dim_vs_delay_t, \
    single_cell_visualization, plot_decode_sample_from_single_time, time_decode_lin_reg
import sklearn

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')
seed = 1

main_dir = '/Users/dongyanlin/Desktop/TUNL_publication/Sci_Reports/data/tunl2d/trained'
data_dir = 'nomem_40_lstm_512_1e-05'
save_dir = os.path.join('/Users/dongyanlin/Desktop/TUNL_publication/Sci_Reports/figure/tunl2d/trained', data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
data = np.load(os.path.join(main_dir, data_dir, data_dir+'_seed_1_epi59999.pt_data.npz'), allow_pickle=True)  # data.npz file

hparams = data_dir.split('_')
env_type = hparams[0]
len_delay = int(hparams[1])
hidden_type = hparams[2]
n_neurons = int(hparams[3])
lr = float(hparams[4])
env_title = 'Mnemonic_TUNL' if env_type == 'mem' else 'Non-mnemonic_TUNL'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

behaviour_only = False  # plot performance data
plot_performance = True
if behaviour_only:
    stim = data['stim']  # n_total_episodes x 2
    choice = data['choice']  # n_total_episodes x 2
    epi_nav_reward = data['epi_nav_reward']  # n_total_episodes
    ideal_nav_reward = data['ideal_nav_rwds']  # n_total_episodes
    n_total_episodes = np.shape(stim)[0]
    nonmatch = np.any(stim != choice, axis=0)
    avg_nav_rewards = bin_rewards(epi_nav_reward, n_total_episodes//10)
    binned_nonmatch_perc = bin_rewards(nonmatch, n_total_episodes//10)
    if plot_performance:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(6, 6))
        fig.suptitle(f'{env_title} TUNL')
        ax1.plot(np.arange(n_total_episodes), avg_nav_rewards, label=net_title)
        ax1.plot(np.arange(n_total_episodes), ideal_nav_reward, label="Ideal navigation reward")
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Navigation reward')
        ax1.legend()

        ax2.plot(np.arange(n_total_episodes), binned_nonmatch_perc, label=net_title)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Fraction Nonmatch')
        ax2.set_ylim(0,1)
        ax2.legend()
        fig.savefig(save_dir + f'/performance.svg')
    # TODO: exit script


stim = data['stim']  # n_total_episodes x 2
choice = data['choice']  # n_total_episodes x 2
delay_loc = data['delay_loc']  # n_total_episodes x len_delay x 2
delay_resp_hx = data['delay_resp_hx']  # n_total_episodes x len_delay x n_neurons
delay_resp_cx = data['delay_resp_cx']  # n_total_episodes x len_delay x n_neurons
epi_nav_reward = data['epi_nav_reward']  # n_total_episodes
ideal_nav_reward = data['ideal_nav_rwds']  # n_total_episodes
n_total_episodes = np.shape(stim)[0]

normalize = True
if normalize:
    delay_resp = sklearn.preprocessing.normalize(np.reshape(delay_resp_hx, (n_total_episodes*len_delay, n_neurons)), axis=0, norm='l1').reshape((n_total_episodes, len_delay, n_neurons))
else:
    delay_resp = delay_resp_hx

# # Select units with large enough variation in its activation
# big_var_neurons = []
# for i_neuron in range(512):
#     if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
#         big_var_neurons.append(i_neuron)
# delay_resp = delay_resp_hx[:, 2:, [x for x in range(512) if x in big_var_neurons]]
# delay_loc = delay_loc[:, 2:, :]

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

binary_nonmatch = np.any(stim != choice, axis=1)
correct_resp = delay_resp[binary_nonmatch == 1]
incorrect_resp = delay_resp[binary_nonmatch == 0]
correct_loc = delay_loc[binary_nonmatch == 1]  # delay locations on correct trials
incorrect_loc = delay_loc[binary_nonmatch == 0]

cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
    delay_resp, norm=True)
nontime_cell_idx = [x for x in range(n_neurons) if (x not in cell_nums_seq and x not in cell_nums_ramp)]
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_seq = delay_resp[:, :, cell_nums_seq]
delay_resp_nontime = delay_resp[:,:,nontime_cell_idx]
left_stim_resp_ramp = delay_resp_ramp[np.all(stim == [1, 1], axis=1)]
right_stim_resp_ramp = delay_resp_ramp[np.any(stim != [1, 1], axis=1)]
left_stim_resp_seq = delay_resp_seq[np.all(stim == [1, 1], axis=1)]
right_stim_resp_seq = delay_resp_seq[np.any(stim != [1, 1], axis=1)]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)
#
print('Make a pie chart of neuron counts...')
make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, save_dir, env_title, save=False)

print('Sort avg resp analysis...')
plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=env_title+' Sequence cells', remove_nan=True, save_dir=save_dir, save=False)
plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=env_title+' Ramping cells', remove_nan=True, save_dir=save_dir, save=False)
plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=env_title+' All cells', remove_nan=True, save_dir=save_dir, save=False)

print('sort in same order analysis...')
plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=env_title+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons, save_dir=save_dir, save=False)
plot_sorted_in_same_order(left_stim_resp_seq, right_stim_resp_seq, 'Left', 'Right', big_title=env_title+' Sequence cells', len_delay=len_delay, n_neurons=n_seq_neurons, save_dir=save_dir, save=False)
plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=env_title+' All cells', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=False)
plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=env_title+' All cells ic', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=False)

print('decode stim analysis...')
plot_decode_sample_from_single_time(delay_resp, binary_stim, env_title+' All Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=False)
plot_decode_sample_from_single_time(delay_resp_ramp, binary_stim, env_title+' Ramping Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=False)
plot_decode_sample_from_single_time(delay_resp_seq, binary_stim, env_title+' Sequence Cells', n_fold=7, max_iter=100, save_dir=save_dir, save=False)

print('decode time analysis...')
time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title=env_title+' All cells', save_dir=save_dir, save=False)
time_decode_lin_reg(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=env_title+' Ramping cells', save_dir=save_dir, save=False)
time_decode_lin_reg(delay_resp_seq, len_delay, n_seq_neurons, 1000, title=env_title+' Sequence cells', save_dir=save_dir, save=False)

print('Single-cell visualization... SAVE AUTOMATICALLY')
single_cell_visualization(delay_resp, binary_stim, cell_nums_ramp, type='ramp', save_dir=save_dir)
single_cell_visualization(delay_resp, binary_stim, cell_nums_seq, type='seq', save_dir=save_dir)


# ==========================================

# Mutual information analysis
# ==========================================

# mutual information analysis

ratemap, spatial_occupancy = construct_ratemap(delay_resp, delay_loc)
mutual_info = calculate_mutual_information(ratemap, spatial_occupancy)
shuffled_mutual_info = calculate_shuffled_mutual_information(delay_resp, delay_loc, n_total_episodes)

plot_mutual_info_distribution(mutual_info, title='all_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info, save_dir=save_dir, save=False)

joint_encoding_info(delay_resp, delay_loc, analysis='selectivity', recalculate=True)
plot_joint_encoding_information(save_dir=save_dir, title='all_cells')

ratemap_left_sti, spatial_occupancy_left_sti = construct_ratemap(left_stim_resp, left_stim_loc)
ratemap_right_sti, spatial_occupancy_right_sti = construct_ratemap(right_stim_resp, right_stim_loc)
mutual_info_left_sti = calculate_mutual_information(ratemap_left_sti, spatial_occupancy_left_sti)
mutual_info_right_sti = calculate_mutual_information(ratemap_right_sti, spatial_occupancy_right_sti)

print("Plot splitter cells... SAVE AUTOMATICALLY")
plot_stimulus_selective_place_cells(mutual_info_left_sti, ratemap_left_sti, mutual_info_right_sti, ratemap_right_sti, save_dir=save_dir)

decode_sample_from_trajectory(delay_loc, stim, save_dir=save_dir, save=False)

print("Mutual info for ramping cells and sequence cells")

# ======= Starting Line 110 of analysis_place.py ==========
ratemap_seq, spatial_occupancy_seq = construct_ratemap(delay_resp_seq, delay_loc)
mutual_info_seq = calculate_mutual_information(ratemap_seq, spatial_occupancy_seq)
shuffled_mutual_info_seq = calculate_shuffled_mutual_information(delay_resp_seq, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_seq, title='seq_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_seq, save_dir=save_dir, save=False)
print("seq: ", np.nanmean(mutual_info_seq))
print("shuffled seq: ", np.nanmean(shuffled_mutual_info_seq))

ratemap_ramp, spatial_occupancy_ramp = construct_ratemap(delay_resp_ramp, delay_loc)
mutual_info_ramp = calculate_mutual_information(ratemap_ramp, spatial_occupancy_ramp)
shuffled_mutual_info_ramp = calculate_shuffled_mutual_information(delay_resp_ramp, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_ramp, title='ramp_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp, save_dir=save_dir, save=False)
print("ramp: ", np.nanmean(mutual_info_ramp))
print("shuffled ramp: ", np.nanmean(shuffled_mutual_info_ramp))

ratemap_nontime, spatial_occupancy_nontime = construct_ratemap(delay_resp_nontime, delay_loc)
mutual_info_nontime = calculate_mutual_information(ratemap_nontime, spatial_occupancy_nontime)
shuffled_mutual_info_nontime = calculate_shuffled_mutual_information(delay_resp_nontime, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_ramp, title='non_time_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp, save_dir=save_dir, save=False)
print("Non-time: ", np.nanmean(mutual_info_nontime))
print("shuffled non-time: ", np.nanmean(shuffled_mutual_info_nontime))


print('Analysis finished')
