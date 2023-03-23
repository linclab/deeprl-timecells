from utils_int_discrim import *
from utils_analysis import time_decode_lin_reg, make_venn_diagram, plot_sorted_averaged_resp
import numpy as np
import os
import argparse
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

parser = argparse.ArgumentParser(description="Head-fixed 1D interval discrimination task simulation")
parser.add_argument("--main_dir",type=str,default='/network/scratch/l/lindongy/timecell/data_collecting/timing',help="main data directory")
parser.add_argument("--data_dir",type=str,default='lstm_128_1e-05',help="directory in which .npz is saved")
parser.add_argument("--main_save_dir", type=str, default='/network/scratch/l/lindongy/timecell/data_analysis/timing', help="main directory in which agent-specific directory will be created")
parser.add_argument("--seed", type=int, help="seed to analyse")
parser.add_argument("--episode", type=int, help="ckpt episode to analyse")
parser.add_argument("--behaviour_only", type=bool, default=False, help="whether the data only includes performance data")
parser.add_argument("--normalize", type=bool, default=True, help="normalize each unit's response by its maximum and minimum")
parser.add_argument("--n_shuffle", type=int, default=100, help="number of shuffles to acquire null distribution")
parser.add_argument("--percentile", type=float, default=99.9, help="P threshold to determind significance")
args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)
main_dir = argsdict['main_dir']
data_dir = argsdict['data_dir']
seed = argsdict['seed']
epi = argsdict['episode']
n_shuffle = argsdict['n_shuffle']
percentile = argsdict['percentile']
if not os.path.exists(os.path.join('/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/agents/timing', data_dir, f'seed_{seed}_epi{epi}.pt')):
    print("agent does not exist. exiting.")
    sys.exit()
if not os.path.exists(os.path.join(main_dir, data_dir, data_dir+f'_seed_{seed}_epi{epi}.pt_data.npz')):
    print("data does not exist. exiting.")
    sys.exit()
agent_str = f"{seed}_{epi}_{n_shuffle}_{percentile}"
save_dir = os.path.join(argsdict['main_save_dir'], data_dir, agent_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
data = np.load(os.path.join(main_dir, data_dir, data_dir+f'_seed_{seed}_epi{epi}.pt_data.npz'), allow_pickle=True)  # data.npz file

hparams = data_dir.split('_')
hidden_type = hparams[0]
n_neurons = int(hparams[1])
lr = float(hparams[2])
if len(hparams) > 3:  # weight_decay or dropout
    if 'wd' in hparams[3]:
        wd = float(hparams[3][2:])
    if 'p' in hparams[3]:
        p = float(hparams[3][1:])
        dropout_type = hparams[4]
env_title = 'Interval_Discrimination_1D'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

behaviour_only = True if argsdict['behaviour_only'] == True or argsdict['behaviour_only'] == 'True' else False  # plot performance data

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

action_hist = data["action_hist"]
correct_trials = data["correct_trial"]
stim = data["stim"]
stim1_resp = data["stim1_resp_hx"]  # Note: this could also be linear activity of Feedforward network
stim2_resp = data["stim2_resp_hx"]  # Note: this could also be linear activity of Feedforward network
delay_resp = data["delay_resp_hx"]  # Note: this could also be linear activity of Feedforward network
n_total_episodes = np.shape(stim)[0]
plot_performance(stim, action_hist, title=env_title, save_dir=save_dir, fig_type='matrix')
plot_performance(stim, action_hist,  title=env_title, save_dir=save_dir, fig_type='curve')

# Select units with large enough variation in its activation
# big_var_neurons = []
# for i_neuron in range(512):
#     if np.ptp(np.concatenate(resp_hx[:, :, i_neuron])) > 0.0000001:
#         big_var_neurons.append(i_neuron)
# stim1_resp = stim1_resp[:, :, [x for x in range(512) if x in big_var_neurons]]
# delay_resp = delay_resp_hx[:, :, [x for x in range(512) if x in big_var_neurons]]
# stim2_resp = stim2_resp[:, :, [x for x in range(512) if x in big_var_neurons]]

normalize = True if argsdict['normalize'] == True or argsdict['normalize'] == 'True' else False
if normalize:
    reshape_resp = np.reshape(stim1_resp, (n_total_episodes*40, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    stim1_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(stim2_resp, (n_total_episodes*40, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    stim2_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(delay_resp, (n_total_episodes*20, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    delay_resp = np.reshape(reshape_resp, (n_total_episodes, 20, n_neurons))

last_step_resp = stim2_resp[np.arange(n_total_episodes), stim[:,1]-1, :]


# Plot cell activities in all combinations of stimulus length

plot_time_cell_sorted_same_order(stim, stim1_resp, stim2_resp,save_dir=save_dir, save=True)  # Plot stim1 and stim2 time cells sorted in same order
single_cell_temporal_tuning(stim, stim1_resp, stim2_resp, save_dir=save_dir, compare_correct=True)  # if compare_correct, then SAVE AUTOMATICALLY
retiming(stim, stim1_resp, stim2_resp, save_dir=save_dir, verbose=True)  # SAVE AUTOMATICALLY. Takes ~10 minutes
linear_readout(stim, stim1_resp, stim2_resp, save_dir=save_dir)  # SAVE AUTOMATICALLY
decoding(stim, last_step_resp, save_dir=save_dir, save=True)  # decodes stim1, stim2, stim1>stim2, and stim1-stim2
manifold(stim, last_step_resp, save_dir=save_dir, save=True)  # splits last step resp into stim1>stim2 and stim1<stim2 conditions
time_decode_all_stim_len(stim1_resp, stim[:,0], title='stim1_resp', save_dir=save_dir, save=True)  # decode time, grouped by stimulus length
time_decode_lin_reg(stim1_resp, 40, n_neurons, 1000, "stim1", save_dir=save_dir, save=True)
time_decode_lin_reg(stim2_resp, 40, n_neurons, 1000, "stim2", save_dir=save_dir, save=True)
time_decode_lin_reg(delay_resp, 20, n_neurons, 1000, "delay", save_dir=save_dir, save=True)

compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, analysis="decoding", resp2=stim2_resp, title='', save_dir=save_dir, save=True)  # decode time; train on corr, test on incorr
compare_correct_vs_incorrect(stim1_resp, stim, correct_trials, analysis="population", resp2=stim2_resp, title='', save_dir=save_dir, save=True)  # plot population sequence for corr and incorr trials
# single_cell_visualization(stim2_resp, correct_trials, np.arange(n_neurons), "")


# =============================================

# Identify time cells and ramping cells
# =============================================
short_trial_resp = stim1_resp[stim[:, 0] <= 25, :25, :]
long_trial_resp = stim1_resp[stim[:, 0] > 25]
plot_r_tuning_curves(short_trial_resp, long_trial_resp, 'short_trial_stim_1', 'long_trial_stim_1', save_dir=save_dir, varying_duration=True)


split_1_idx = np.arange(start=0, stop=len(stim1_resp)-1, step=2)
split_2_idx = np.arange(start=1, stop=len(stim1_resp), step=2)
odd_trial_resp = stim1_resp[split_2_idx]
even_trial_resp = stim1_resp[split_1_idx]
plot_r_tuning_curves(odd_trial_resp, even_trial_resp, 'odd_trial_stim_1', 'even_trial_stim_1', save_dir=save_dir, varying_duration=True)

short_trial_resp = stim2_resp[stim[:, 0] <= 25, :25, :]
long_trial_resp = stim2_resp[stim[:, 0] > 25]
plot_r_tuning_curves(short_trial_resp, long_trial_resp, 'short_trial_stim_2', 'long_trial_stim_2', save_dir=save_dir, varying_duration=True)

split_1_idx = np.arange(start=0, stop=len(stim2_resp)-1, step=2)
split_2_idx = np.arange(start=1, stop=len(stim2_resp), step=2)
odd_trial_resp = stim2_resp[split_2_idx]
even_trial_resp = stim2_resp[split_1_idx]
plot_r_tuning_curves(odd_trial_resp, even_trial_resp, 'odd_trial_stim_2', 'even_trial_stim_2', save_dir=save_dir, varying_duration=True)

plot_r_tuning_curves(stim1_resp, stim2_resp, 'stim_1', 'stim_2', save_dir=save_dir, varying_duration=True)

#=================
# Define period that we want to analyse
for (resp, stimulus, label) in zip([stim1_resp,stim2_resp, delay_resp], [stim[:,0],stim[:,1], None], ['stimulus_1', 'stimulus_2', 'delay']):
    print(f"analysing data from {label}")
    ramp_ident_results = np.load(os.path.join(os.path.join(argsdict['main_save_dir'], data_dir),f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_ramp_ident_results.npz'), allow_pickle=True)
    time_ident_results = np.load(os.path.join(os.path.join(argsdict['main_save_dir'], data_dir),f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_time_cell_results.npz'), allow_pickle=True)
    cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
    cell_nums_time = time_ident_results['time_cell_nums']

    cell_nums_nontime = np.delete(np.arange(n_neurons), np.intersect1d(cell_nums_time, cell_nums_ramp))

    # Here, the cell_nums have not been re-arranged according to peak latency yet
    resp_ramp = resp[:, :, cell_nums_ramp]
    resp_time = resp[:, :, cell_nums_time]
    resp_nontime = resp[:, :, cell_nums_nontime]
    n_ramp_neurons = len(cell_nums_ramp)
    n_time_neurons = len(cell_nums_time)
    n_nontime_neurons = len(cell_nums_nontime)
    
    # See if time cell/ramping cell/nontime cells can decode time
    if label != 'delay':
        time_decode_all_stim_len(resp_ramp, stimulus, title=label+' ramp', save_dir=save_dir, save=True)  # decode time, grouped by stimulus length
        time_decode_all_stim_len(resp_time, stimulus, title=label+' time', save_dir=save_dir, save=True)
        time_decode_all_stim_len(resp_nontime, stimulus, title=label+' nontime', save_dir=save_dir, save=True)
    time_decode_lin_reg(resp_ramp, np.shape(resp)[1], n_ramp_neurons, 1000, title=label+' ramp', save_dir=save_dir, save=True)
    time_decode_lin_reg(resp_time, np.shape(resp)[1], n_time_neurons, 1000, title=label+' time', save_dir=save_dir, save=True)
    time_decode_lin_reg(resp_nontime, np.shape(resp)[1], n_nontime_neurons, 1000, title=label+' nontime', save_dir=save_dir, save=True)
    
    
    # Re-arrange cell_nums according to peak latency
    cell_nums_all, cell_nums_time, cell_nums_ramp, cell_nums_nontime, \
    sorted_matrix_all, sorted_matrix_time, sorted_matrix_ramp, sorted_matrix_nontime = \
        sort_response_by_peak_latency(resp, cell_nums_ramp, cell_nums_time, norm=True)
    
    # print('Make a venn diagram of neuron counts...')
    make_venn_diagram(cell_nums_ramp, cell_nums_time, n_neurons, label=label, save_dir=save_dir, save=True)

    plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=label+"_all_cell_activity", remove_nan=True, save_dir=save_dir, save=True)
    plot_sorted_averaged_resp(cell_nums_time, sorted_matrix_time, title=label+"_time_cell_activity", remove_nan=True, save_dir=save_dir, save=True)
    plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=label+"_ramp_cell_activity", remove_nan=True, save_dir=save_dir, save=True)
    plot_sorted_averaged_resp(cell_nums_nontime, sorted_matrix_nontime, title=label+"_nontime_cell_activity", remove_nan=True, save_dir=save_dir, save=True)

    if label == 'delay':  # this function only makes sense if the trial durations are the same
        plot_field_width_vs_peak_time(resp[:, :, cell_nums_time], save_dir, title='delay_time_cells')
