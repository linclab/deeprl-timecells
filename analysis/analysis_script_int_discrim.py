from utils_int_discrim import *
from analysis_utils import single_cell_visualization, time_decode, time_decode_lin_reg, make_venn_diagram
from analysis.linclab_utils import plot_utils
import numpy as np
import os
import sklearn
import sys

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

main_dir = '/Users/dongyanlin/Desktop/TUNL_publication/Sci_Reports/data/timing/trained'
data_dir = 'lstm_512_5e-06'
save_dir = os.path.join(main_dir.replace('data', 'figure'), data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
data = np.load(os.path.join(main_dir, data_dir, data_dir+'_seed_1_epi59999.pt_data.npz'), allow_pickle=True)  # data.npz file

hparams = data_dir.split('_')
hidden_type = hparams[0]
n_neurons = int(hparams[1])
lr = float(hparams[2])
env_title = 'Interval_Discrimination'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'


behavioral_analysis = False
lesion_experiment = False

# ------------------------------ Behavioral analysis ----------------------------------
behaviour_only = False  # plot performance data
if behaviour_only:
    action_hist = data["action_hist"]
    correct_trials = data["correct_trial"]
    stim = data["stim"]
    n_total_episodes = np.shape(stim)[0]
    # plot_training_performance()
    plot_training_accuracy(stim, action_hist, title=env_title, save_dir=save_dir, save=False, base_episode=0)
    plot_performance(stim, action_hist, title=env_title, save_dir=save_dir, fig_type='matrix')
    plot_performance(stim, action_hist,  title=env_title, save_dir=save_dir, fig_type='curve')
    # plot_training_performance() # TODO: fix this function
    sys.exit()
else:
    action_hist = data["action_hist"]
    correct_trials = data["correct_trial"]
    stim = data["stim"]
    stim1_resp = data["stim1_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    stim2_resp = data["stim2_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    delay_resp = data["delay_resp_hx"]  # Note: this could also be linear activity of Feedforward network
    n_total_episodes = np.shape(stim)[0]
    plot_performance(stim, action_hist, title=env_title, save_dir=save_dir, fig_type='matrix')
    plot_performance(stim, action_hist,  title=env_title, save_dir=save_dir, fig_type='curve')
    sys.exit()

# Select units with large enough variation in its activation
# big_var_neurons = []
# for i_neuron in range(512):
#     if np.ptp(np.concatenate(resp_hx[:, :, i_neuron])) > 0.0000001:
#         big_var_neurons.append(i_neuron)
# stim1_resp = stim1_resp[:, :, [x for x in range(512) if x in big_var_neurons]]
# delay_resp = delay_resp_hx[:, :, [x for x in range(512) if x in big_var_neurons]]
# stim2_resp = stim2_resp[:, :, [x for x in range(512) if x in big_var_neurons]]

normalize = False
if normalize:
    reshape_resp = np.reshape(stim1_resp, (n_total_episodes*40, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=1, keepdims=True)) / np.ptp(reshape_resp, axis=1, keepdims=True)
    stim1_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(stim2_resp, (n_total_episodes*ld, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=1, keepdims=True)) / np.ptp(reshape_resp, axis=1, keepdims=True)
    stim2_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(delay_resp, (n_total_episodes*20, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=1, keepdims=True)) / np.ptp(reshape_resp, axis=1, keepdims=True)
    delay_resp = np.reshape(reshape_resp, (n_total_episodes, 20, n_neurons))

resp_hx = np.concatenate((stim1_resp, delay_resp, stim2_resp), axis=1)
last_step_resp = stim2_resp[np.arange(n_total_episodes), stim[:,1]-1, :]

# Plot cell activities in all combinations of stimulus length

plot_time_cell_sorted_same_order(stim, stim1_resp, stim2_resp,save_dir=save_dir, save=False)  # Plot stim1 and stim2 time cells sorted in same order
single_cell_temporal_tuning(stim, stim1_resp, stim2_resp, save_dir=save_dir, compare_correct=False)  # if compare_correct, then SAVE AUTOMATICALLY
retiming(stim, stim1_resp, stim2_resp, save_dir=save_dir, verbose=True)  # SAVE AUTOMATICALLY. Takes ~10 minutes
linear_readout(stim, stim1_resp, stim2_resp, save_dir=save_dir)  # SAVE AUTOMATICALLY
decoding(stim, last_step_resp, save_dir=save_dir, save=False)  # decodes stim1, stim2, stim1>stim2, and stim1-stim2
manifold(stim, last_step_resp, save_dir=save_dir, save=False)  # splits last step resp into stim1>stim2 and stim1<stim2 conditions
time_decode_all_stim_len(stim1_resp, stim[:,0], title='stim1_resp', save_dir=save_dir, save=False)  # decode time, grouped by stimulus length
time_decode_lin_reg(stim1_resp, 40, n_neurons, 1000, "stim1", save_dir=save_dir, save=False)
time_decode_lin_reg(stim2_resp, 40, n_neurons, 1000, "stim2", save_dir=save_dir, save=False)
time_decode_lin_reg(delay_resp, 20, n_neurons, 1000, "delay", save_dir=save_dir, save=False)
compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, analysis="decoding", resp2=stim2_resp, title='', save_dir=save_dir, save=False)  # decode time; train on corr, test on incorr
compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, analysis="population", resp2=stim2_resp, title='', save_dir=save_dir, save=False)  # plot population sequence for corr and incorr trials
# single_cell_visualization(stim2_resp, correct_trials, np.arange(n_neurons), "")


# =============================================

# Identify time cells and ramping cells
# =============================================

# Define period that we want to analyse
resp = stim1_resp  # Or stim2_resp or delay_resp
stimulus = stim[:, 0]  # or stim[:, 1]
label = 'stimulus_1'
# Identifying ramping cells
p_result, slope_result, intercept_result, R_result = lin_reg_ramping(resp)
ramp_cell_bool = np.logical_and(p_result<=0.05, slope_result<=0.05)
cell_nums_ramp = np.where(ramp_cell_bool)[0]

# Identifying sequence cells
RB_result, z_RB_threshold_result = ridge_to_background(resp, ramp_cell_bool, percentile=95, n_shuff=1000)
seq_cell_bool = RB_result > z_RB_threshold_result
cell_nums_seq = np.where(seq_cell_bool)[0]

if np.shape(resp)[1] == 20:  # Delay period with consistent trial duration
    I_result, I_threshold_result = skaggs_temporal_information(resp, n_shuff=1000, percentile=95)
else:
    I_result, I_threshold_result = skaggs_temporal_information_varying_duration(resp, stim, n_shuff=1000, percentile=95)
high_temporal_info_cell_bool = I_result > I_threshold_result

R_result, pval_result = trial_consistency_across_durations(stim, stim1_resp, stim2_resp, type='absolute')  # also try 'relative'

cell_nums_nontime = np.remove(np.arange(n_neurons), np.intersect1d(cell_nums_seq, cell_nums_ramp))


# Here, the cell_nums have not been re-arranged according to peak latency yet
resp_ramp = resp[:, :, cell_nums_ramp]
resp_seq = resp[:, :, cell_nums_seq]
resp_nontime = resp[:, :, cell_nums_nontime]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)
n_nontime_neurons = len(cell_nums_nontime)


# See if time cell/ramping cell/nontime cells can decode time
time_decode_all_stim_len(resp[cell_nums_ramp], stimulus, title=label+' ramp', save_dir=save_dir, save=False)  # decode time, grouped by stimulus length
time_decode_all_stim_len(resp[cell_nums_seq], stimulus, title=label+' seq', save_dir=save_dir, save=False)
time_decode_all_stim_len(resp[cell_nums_nontime], stimulus, title=label+' nontime', save_dir=save_dir, save=False)
time_decode_lin_reg(resp[cell_nums_ramp], np.shape(resp[1]), n_ramp_neurons, 1000, title=label+' ramp', save_dir=save_dir, save=False)
time_decode_lin_reg(resp[cell_nums_seq], np.shape(resp[1]), n_seq_neurons, 1000, title=label+' seq', save_dir=save_dir, save=False)
time_decode_lin_reg(resp[cell_nums_nontime], np.shape(resp[1]), n_nontime_neurons, 1000, title=label+' nontime', save_dir=save_dir, save=False)


# Re-arrange cell_nums according to peak latency
cell_nums_all, cell_nums_seq, cell_nums_ramp, cell_nums_nontime, \
sorted_matrix_all, sorted_matrix_seq, sorted_matrix_ramp, sorted_matrix_nontime = \
    sort_response_by_peak_latency(resp, cell_nums_ramp, cell_nums_seq, norm=True)

# print('Make a venn diagram of neuron counts...')
make_venn_diagram(cell_nums_ramp, cell_nums_seq, n_neurons, title=label, save_dir=save_dir, save=False)


plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=label+"_all_cell_activity", remove_nan=True, save_dir=save_dir, save=False)
plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=label+"_seq_cell_activity", remove_nan=True, save_dir=save_dir, save=False)
plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=label+"_ramp_cell_activity", remove_nan=True, save_dir=save_dir, save=False)
plot_sorted_averaged_resp(cell_nums_nontime, sorted_matrix_nontime, title=label+"_nontime_cell_activity", remove_nan=True, save_dir=save_dir, save=False)


# ------------------------------ Lesion experiment ----------------------------------

if lesion_experiment:
    data = np.load('data/lesion.npz')
    postlesion_acc = data['postlesion_perf']
    #plot_postlesion_performance(postlesion_acc, compare="lesion type")
    plot_postlesion_performance(postlesion_acc, compare="stimulus length")