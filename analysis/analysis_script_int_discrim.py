from utils_int_discrim import *
from analysis.linclab_utils import plot_utils
import numpy as np

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

behavioral_analysis = False
hidden_activity_analysis = True
lesion_experiment = False
load_processed_data = False
data_path = 'data/ccn/2022_05_23_16_11_53_Interval_Discrimination_lstm/'
label = 'Interval Discrimination'

# ------------------------------ Load data ----------------------------------

if load_processed_data:
    data = np.load(data_path+'data_processed.npz')
    stim = data["stim"]
    stim1_resp = data["stim1_resp"]
    stim2_resp = data["stim2_resp"]
    delay_resp = data["delay_resp"]
    last_step_resp = data["last_step_resp"]
else:
    data = np.load(data_path+'data.npz')
    keep_episode = -20000
    correct_trials = data["correct_trial"][keep_episode:]
    stim = data["stim"][keep_episode:, :]
    stim1_resp_hx = data["stim1_resp_hx"][keep_episode:, :, :]
    stim2_resp_hx = data["stim2_resp_hx"][keep_episode:, :, :]
    delay_resp_hx = data["delay_resp_hx"][keep_episode:, :, :]

stim_set = np.sort(np.unique(stim))
num_stim = np.max(np.shape(stim_set))
n_episodes = np.shape(stim)[0]
n_neurons = 512
figpath = 'data/'

# ------------------------------ Behavioral analysis ----------------------------------
# if behavioral_analysis:
#     plot_training_accuracy(stim, action_hist, base_episode=90000)
#     plot_performance(stim, action_hist, type='matrix')
#     plot_performance(stim, action_hist, type='curve')
#     print("Behavioral_analysis finished")

# ------------------------------ Hidden unit activity analysis ----------------------------------

if hidden_activity_analysis:
    if not load_processed_data:
        # Select units with large enough variation in its activation
        resp_hx = np.concatenate((stim1_resp_hx, delay_resp_hx, stim2_resp_hx), axis=1)
        big_var_neurons = []
        for i_neuron in range(512):
            if np.ptp(np.concatenate(resp_hx[:, :, i_neuron])) > 0.0000001:
                big_var_neurons.append(i_neuron)
        stim1_resp = stim1_resp_hx[:, :, [x for x in range(512) if x in big_var_neurons]]
        delay_resp = delay_resp_hx[:, :, [x for x in range(512) if x in big_var_neurons]]
        stim2_resp = stim2_resp_hx[:, :, [x for x in range(512) if x in big_var_neurons]]
        last_step_resp = stim2_resp[np.arange(n_episodes), stim[:,1]-1, :]
        n_neurons = np.shape(stim1_resp)[-1]

    # Plot cell activities in all combinations of stimulus length
    #plot_time_cell(stim[:,0], stim1_resp, label="First_Stim_", figpath='data/analysis/population/stim1/')
    #plot_time_cell_sorted_same_order(stim, stim1_resp, stim2_resp)
    #single_cell_temporal_tuning(stim, stim1_resp, stim2_resp)
    #retiming(stim, stim1_resp, stim2_resp)
    #linear_readout(stim, stim1_resp, stim2_resp)
    #decoding(stim, last_step_resp, data_path)
    #manifold(stim, last_step_resp)
    #time_decode_all_stim_len(stim1_resp, stim[:,0])
    #plot_training_performance()
    #time_decode(stim1_resp, 40, 512, 1000, "Time_decoding", plot=True)
    compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, analysis="decoding", resp2=stim2_resp)
    #compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, resp2=stim2_resp)
    #single_cell_visualization(stim2_resp, correct_trials, np.arange(n_neurons), "")

    # analyze delay period activity
    #cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
    #delay_resp, norm=True)
    #plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title="Delay phase population activity", remove_nan=True)
    #time_decode(delay_resp, np.shape(delay_resp)[1], n_neurons, 1000, title="Delay phase", plot=True)

# ------------------------------ Lesion experiment ----------------------------------

if lesion_experiment:
    data = np.load('data/lesion.npz')
    postlesion_acc = data['postlesion_perf']
    #plot_postlesion_performance(postlesion_acc, compare="lesion type")
    plot_postlesion_performance(postlesion_acc, compare="stimulus length")