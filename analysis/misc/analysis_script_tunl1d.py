from utils_time_ramp import *
from utils_analysis import *
#from expts.envs.tunl_1d import *
import sys
import argparse
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")


def single_cell_plot(total_resp, binary_stim, i_neuron, start_trial):
    assert start_trial+100 <= sum(binary_stim == 0) and start_trial+100 <= sum(binary_stim == 1), "start_trial exceeds number of left or right trials"
    xl = total_resp[binary_stim == 0, :, i_neuron][start_trial:start_trial+100]
    xr = total_resp[binary_stim == 1, :, i_neuron][start_trial:start_trial+100]
    norm_xl = stats.zscore(xl, axis=1)
    norm_xr = stats.zscore(xr, axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(5, 8), sharex='all',
                                        gridspec_kw={'height_ratios': [2, 2, 1.5]})
    fig.suptitle(f'Unit #{i_neuron}')

    im = ax1.imshow(norm_xl, cmap='jet')
    ax1.set_aspect('auto')
    ax1.set_xticks(np.arange(len_delay, step=10))
    ax1.set_yticks([0, len(norm_xl)])
    ax1.set_yticklabels(['1', '100'])
    ax1.set_ylabel(f'Left trials')

    im2 = ax2.imshow(norm_xr, cmap='jet')
    ax2.set_aspect('auto')
    ax2.set_xticks(np.arange(len_delay, step=10))
    ax2.set_yticks([0, len(norm_xr)])
    ax2.set_yticklabels(['1', '100'])
    ax2.set_ylabel(f'Right trials')

    ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xl, axis=0), axis=0), label='Left', color=utils_linclab_plot.LINCLAB_COLS['yellow'])
    ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xr, axis=0), axis=0), label='Right', color=utils_linclab_plot.LINCLAB_COLS['brown'])
    ax3.set_xlabel('Time since delay period onset')
    ax3.legend(loc='upper right', fontsize='medium')
    ax3.set_ylabel('Avg activation')
    plt.show()

parser = argparse.ArgumentParser(description="Head-fixed 1D TUNL task simulation")
parser.add_argument("--main_dir",type=str,default='/network/scratch/l/lindongy/timecell/data_collecting/tunl1d_og',help="main data directory")
parser.add_argument("--data_dir",type=str,default='mem_40_lstm_256_0.0001_p0.25_2',help="directory in which .npz is saved")
parser.add_argument("--main_save_dir", type=str, default='/network/scratch/l/lindongy/timecell/data_analysis/tunl1d_og', help="main directory in which agent-specific directory will be created")
parser.add_argument("--seed", type=int, help="seed to analyse")
parser.add_argument("--episode", type=int, help="ckpt episode to analyse")
parser.add_argument("--behaviour_only", type=bool, default=False, help="whether the data only includes performance data")
parser.add_argument("--plot_performance", type=bool, default=True,  help="if behaviour only, whether to plot the performance plot")
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
agent_str = f"{seed}_{epi}_{n_shuffle}_{percentile}"
save_dir = os.path.join(argsdict['main_save_dir'], data_dir, agent_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
data = np.load(os.path.join(main_dir, data_dir, data_dir+f'_seed_{seed}_epi{epi}.pt_data.npz'), allow_pickle=True)  # data.npz file

hparams = data_dir.split('_')
env_type = hparams[0]
len_delay = int(hparams[1])
hidden_type = hparams[2]
n_neurons = int(hparams[3])
lr = float(hparams[4])
if len(hparams) > 5:  # weight_decay or dropout
    if 'wd' in hparams[5]:
        wd = float(hparams[5][2:])
    if 'p' in hparams[5]:
        p = float(hparams[5][1:])
        dropout_type = hparams[6]
env_title = 'Mnemonic_TUNL_1D' if env_type == 'mem' else 'Non-mnemonic_TUNL_1D'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

behaviour_only = True if argsdict['behaviour_only'] == True or argsdict['behaviour_only'] == 'True' else False  # plot performance data
plot_performance = True if argsdict['plot_performance'] == True or argsdict['plot_performance'] == 'True' else False
if behaviour_only:
    stim = data['stim']  # n_total_episodes
    first_action = data['first_action']  # n_total_episodes
    n_total_episodes = np.shape(stim)[0]
    nonmatch = stim != first_action
    binned_nonmatch_perc = bin_rewards(nonmatch, n_total_episodes//10)
    if plot_performance:
        fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
        ax2.plot(np.arange(n_total_episodes), binned_nonmatch_perc, label=net_title)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Fraction Nonmatch')
        ax2.set_ylim(0,1)
        ax2.legend()
        fig.savefig(save_dir + f'/performance.svg')
    sys.exit()


stim = data['stim']  # n_total_episodes
first_action = data['first_action']  # n_total_episodes
delay_resp = data['delay_resp']  # n_total_episodes x len_delay x n_neurons
n_total_episodes = np.shape(stim)[0]

normalize = True if argsdict['normalize'] == True or argsdict['normalize'] == 'True' else False
if normalize:
    reshape_resp = np.reshape(delay_resp, (n_total_episodes*len_delay, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    delay_resp = np.reshape(reshape_resp, (n_total_episodes, len_delay, n_neurons))


# # Select units with large enough variation in its activation
# big_var_neurons = []
# for i_neuron in range(512):
#     if np.ptp(np.concatenate(delay_resp_hx[:, :, i_neuron])) > 0.0000001:
#         big_var_neurons.append(i_neuron)
# delay_resp = delay_resp_hx[:, 2:, [x for x in range(512) if x in big_var_neurons]]
# delay_loc = delay_loc[:, 2:, :]

# separate left and right trials
left_stim_resp = delay_resp[stim == 0]
right_stim_resp = delay_resp[stim == 1]

left_choice_resp = delay_resp[first_action == 0]
right_choice_resp = delay_resp[first_action == 1]

binary_nonmatch = np.ones(n_total_episodes)
binary_nonmatch[stim == first_action] = 0  # 0 is match, 1 is nonmatch

correct_resp = delay_resp[binary_nonmatch == 1]
incorrect_resp = delay_resp[binary_nonmatch == 0]

# cell_nums_all, sorted_matrix_all, cell_nums_time, sorted_matrix_time, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_time(
#     delay_resp, norm=True)

# tuning_curve_dim_reduction(left_stim_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_left')
# tuning_curve_dim_reduction(right_stim_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_right')
# tuning_curve_dim_reduction(delay_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_all')
# breakpoint()
# plot left-right pearson r
plot_r_tuning_curves(left_stim_resp, right_stim_resp, 'left', 'right', save_dir=save_dir)

for resp, label in zip([left_stim_resp, right_stim_resp], ['left', 'right']):
    split_1_idx = np.arange(start=0, stop=len(resp)-1, step=2)
    split_2_idx = np.arange(start=1, stop=len(resp), step=2)
    odd_trial_resp = resp[split_2_idx]
    even_trial_resp = resp[split_1_idx]
    plot_r_tuning_curves(odd_trial_resp, even_trial_resp, f"{label}_odd", f"{label}_even", save_dir=save_dir)

ramp_ident_results = np.load(os.path.join(os.path.join(argsdict['main_save_dir'], data_dir),f'{seed}_{epi}_{n_shuffle}_{percentile}_ramp_ident_results.npz'), allow_pickle=True)
time_ident_results = np.load(os.path.join(os.path.join(argsdict['main_save_dir'], data_dir),f'{seed}_{epi}_{n_shuffle}_{percentile}_time_cell_results.npz'), allow_pickle=True)
cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
cell_nums_ramp_l = ramp_ident_results['cell_nums_ramp_l']
cell_nums_ramp_r = ramp_ident_results['cell_nums_ramp_r']
cell_nums_time = time_ident_results['time_cell_nums']
cell_nums_time_l = time_ident_results['time_cell_nums_l']
cell_nums_time_r = time_ident_results['time_cell_nums_l']


plot_field_width_vs_peak_time(left_stim_resp[:, :, cell_nums_time_l], save_dir=save_dir, title='left_time_cells')
plot_field_width_vs_peak_time(right_stim_resp[:, :, cell_nums_time_r], save_dir=save_dir, title='right_time_cells')

cell_nums_nontime = np.delete(np.arange(n_neurons), np.intersect1d(cell_nums_time, cell_nums_ramp))


# Here, the cell_nums have not been re-arranged according to peak latency yet
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_time = delay_resp[:, :, cell_nums_time]
delay_resp_nontime = delay_resp[:, :, cell_nums_nontime]
left_stim_resp_ramp = delay_resp_ramp[stim == 0]
right_stim_resp_ramp = delay_resp_ramp[stim == 1]
left_stim_resp_time = delay_resp_time[stim == 0]
right_stim_resp_time = delay_resp_time[stim == 1]
left_stim_resp_nontime = delay_resp_nontime[stim == 0]
right_stim_resp_nontime = delay_resp_nontime[stim == 1]
n_ramp_neurons = len(cell_nums_ramp)
n_time_neurons = len(cell_nums_time)
n_nontime_neurons = len(cell_nums_nontime)

# Re-arrange cell_nums according to peak latency
cell_nums_all, cell_nums_time, cell_nums_ramp, cell_nums_nontime, \
sorted_matrix_all, sorted_matrix_time, sorted_matrix_ramp, sorted_matrix_nontime = \
    sort_response_by_peak_latency(delay_resp, cell_nums_ramp, cell_nums_time, norm=True)

# print('Make a venn diagram of neuron counts...')
make_venn_diagram(cell_nums_ramp, cell_nums_time, n_neurons, label=f'{seed}_{epi}', save_dir=save_dir, save=True)

print('Sort avg resp analysis...')
plot_sorted_averaged_resp(cell_nums_time, sorted_matrix_time, title=env_title+' Time cells', remove_nan=True, save_dir=save_dir, save=True)
plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=env_title+' Ramping cells', remove_nan=True, save_dir=save_dir, save=True)
plot_sorted_averaged_resp(cell_nums_nontime, sorted_matrix_nontime, title=env_title+' Non-temporal cells', remove_nan=True, save_dir=save_dir, save=True)
plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=env_title+' All cells', remove_nan=True, save_dir=save_dir, save=True)

print('sort in same order analysis...')
plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=env_title+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons, save_dir=save_dir, save=True)
plot_sorted_in_same_order(left_stim_resp_time, right_stim_resp_time, 'Left', 'Right', big_title=env_title+' Time cells', len_delay=len_delay, n_neurons=n_time_neurons, save_dir=save_dir, save=True)
plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=env_title+' All cells', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=True)
#plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=env_title+' All cells ic', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=False)

print('decode stim analysis...')
plot_decode_sample_from_single_time(delay_resp, stim, env_title+' All Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=True)
plot_decode_sample_from_single_time(delay_resp_ramp, stim, env_title+' Ramping Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=True)
plot_decode_sample_from_single_time(delay_resp_time, stim, env_title+' Time Cells', n_fold=7, max_iter=100, save_dir=save_dir, save=True)

print('decode time analysis...')
time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title=env_title+' All cells', save_dir=save_dir, save=True)
time_decode_lin_reg(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=env_title+' Ramping cells', save_dir=save_dir, save=True)
time_decode_lin_reg(delay_resp_time, len_delay, n_time_neurons, 1000, title=env_title+' Time cells', save_dir=save_dir, save=True)

print('Single-cell visualization... SAVE AUTOMATICALLY')
single_cell_visualization(delay_resp, stim, cell_nums_ramp, type='ramp', save_dir=save_dir)
single_cell_visualization(delay_resp, stim, cell_nums_time, type='time', save_dir=save_dir)

print('Analysis finished')