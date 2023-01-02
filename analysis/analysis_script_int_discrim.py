from utils_int_discrim import *
from analysis_utils import single_cell_visualization, time_decode, time_decode_lin_reg
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

plot_time_cell(stim[:,0], stim1_resp, label="First_Stim_", save_dir=save_dir, save=False)  # Plots time cell sequences and pie charts
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

# analyze delay period activity
cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
delay_resp, norm=True)
plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title="delay_phase_population_activity", remove_nan=True, save_dir=save_dir, save=False)

# ------------------------------ Lesion experiment ----------------------------------

if lesion_experiment:
    data = np.load('data/lesion.npz')
    postlesion_acc = data['postlesion_perf']
    #plot_postlesion_performance(postlesion_acc, compare="lesion type")
    plot_postlesion_performance(postlesion_acc, compare="stimulus length")