import utils_linclab_plot
from utils_mutual_info import *
from utils_time_ramp import *
from utils_analysis import bin_rewards, make_venn_diagram, plot_sorted_averaged_resp, plot_sorted_in_same_order, \
    single_cell_visualization, plot_decode_sample_from_single_time, time_decode_lin_reg
import sys
import argparse

utils_linclab_plot.linclab_plt_defaults()
utils_linclab_plot.set_font(font='Helvetica')

parser = argparse.ArgumentParser(description="Non-location-fixed 2D TUNL task simulation")
parser.add_argument("--main_dir",type=str,default='/network/scratch/l/lindongy/timecell/data_collecting/tunl2d',help="main data directory")
parser.add_argument("--data_dir",type=str,default='mem_40_lstm_256_1e-05',help="directory in which .npz is saved")
parser.add_argument("--main_save_dir", type=str, default='/network/scratch/l/lindongy/timecell/data_analysis/tunl2d', help="main directory in which agent-specific directory will be created")
parser.add_argument("--seed", type=int, help="seed to analyse")
parser.add_argument("--episode", type=int, help="ckpt episode to analyse")
parser.add_argument("--behaviour_only", type=bool, default=False, help="whether the data only includes performance data")
parser.add_argument("--plot_performance", type=bool, default=True,  help="if behaviour only, whether to plot the performance plot")
parser.add_argument("--normalize", type=bool, default=True, help="normalize each unit's response by its maximum and minimum")
parser.add_argument("--separate_trial_types", type=bool, default=True,  help="identify ramp and sequence cells separately for left-stim and right-stim trials and then combine")
parser.add_argument("--n_shuffle", type=int, default=1000, help="number of shuffles to acquire null distribution")
parser.add_argument("--percentile", type=float, default=95.0, help="P threshold to determind significance")
parser.add_argument("--load_time_ramp_results", type=bool, default=True, help="if true, make sure to have results from time_ramp_ident in save_dir")
args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)
main_dir = argsdict['main_dir']
data_dir = argsdict['data_dir']
save_dir = os.path.join(argsdict['main_save_dir'], data_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
seed = argsdict['seed']
epi = argsdict['episode']
n_shuffle = argsdict['n_shuffle']
percentile = argsdict['percentile']
load_time_ramp_results = True if argsdict['load_time_ramp_results'] == True or argsdict['load_time_ramp_results'] == 'True' else False
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
env_title = 'Mnemonic_TUNL_2D' if env_type == 'mem' else 'Non-mnemonic_TUNL_2D'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

behaviour_only = True if argsdict['behaviour_only'] == True or argsdict['behaviour_only'] == 'True' else False  # plot performance data
plot_performance = True if argsdict['plot_performance'] == True or argsdict['plot_performance'] == 'True' else False
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
        fig.suptitle(f'{env_title} TUNL 2D')
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
    sys.exit()

stim = data['stim']  # n_total_episodes x 2
choice = data['choice']  # n_total_episodes x 2
delay_loc = data['delay_loc']  # n_total_episodes x len_delay x 2
delay_resp_hx = data['delay_resp_hx']  # n_total_episodes x len_delay x n_neurons
delay_resp_cx = data['delay_resp_cx']  # n_total_episodes x len_delay x n_neurons
epi_nav_reward = data['epi_nav_reward']  # n_total_episodes
ideal_nav_reward = data['ideal_nav_rwds']  # n_total_episodes
n_total_episodes = np.shape(stim)[0]
delay_resp = delay_resp_hx

normalize = True if argsdict['normalize'] == True or argsdict['normalize'] == 'True' else False
if normalize:
    reshape_resp = np.reshape(delay_resp, (n_total_episodes*len_delay, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=1, keepdims=True)) / np.ptp(reshape_resp, axis=1, keepdims=True)
    delay_resp = np.reshape(reshape_resp, (n_total_episodes, len_delay, n_neurons))


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

# cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(
#     delay_resp, norm=True)

# tuning_curve_dim_reduction(left_stim_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_left')
# tuning_curve_dim_reduction(right_stim_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_right')
# tuning_curve_dim_reduction(delay_resp, mode='tsne', save_dir=save_dir, title=f'{seed}_{epi}_all')
# breakpoint()


separate_trial_types = True if argsdict['separate_trial_types'] == True or argsdict['separate_trial_types'] == 'True' else False

if load_time_ramp_results:
    if separate_trial_types:
        ramp_ident_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_ramp_ident_results_separate.npz'), allow_pickle=True)
        seq_ident_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_seq_ident_results_separate.npz'), allow_pickle=True)
        trial_reliability_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_trial_reliability_results_separate.npz'), allow_pickle=True)
        temporal_info_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_temporal_info_results_separate.npz'), allow_pickle=True)
    else:
        ramp_ident_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_ramp_ident_results_combined.npz'), allow_pickle=True)
        seq_ident_results = np.load(os.path.join(save_dir, f'{seed}_{epi}_seq_ident_results_combined.npz'), allow_pickle=True)
        trial_reliability_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_trial_reliability_results_combined.npz'), allow_pickle=True)
        temporal_info_results = np.load(os.path.join(save_dir,f'{seed}_{epi}_temporal_info_results_combined.npz'), allow_pickle=True)
    cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
    cell_nums_seq = seq_ident_results['cell_nums_seq']

else: # Run time cell and ramping cell identification script
    # Identifying ramping cells
    if separate_trial_types:
        p_result_l, slope_result_l, intercept_result_l, R_result_l = lin_reg_ramping(left_stim_resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_left')
        ramp_cell_bool_l = np.logical_and(p_result_l<=0.05, np.abs(R_result_l)>=0.9)
        cell_nums_ramp_l = np.where(ramp_cell_bool_l)[0]
        p_result_r, slope_result_r, intercept_result_r, R_result_r = lin_reg_ramping(right_stim_resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_right')
        ramp_cell_bool_r = np.logical_and(p_result_r<=0.05, np.abs(R_result_r)>=0.9)
        cell_nums_ramp_r = np.where(ramp_cell_bool_r)[0]
        ramp_cell_bool = np.logical_or(ramp_cell_bool_l, ramp_cell_bool_r)
        cell_nums_ramp = np.where(ramp_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_ramp_ident_results_separate.npz'),
                            p_result_l=p_result_l, slope_result_l=slope_result_l, intercept_result_l=intercept_result_l, R_result_l=R_result_l,
                            p_result_r=p_result_r,slope_result_r=slope_result_r, intercept_result_r=intercept_result_r, R_result_r=R_result_r,
                            ramp_cell_bool_l=ramp_cell_bool_l,cell_nums_ramp_l=cell_nums_ramp_l,
                            ramp_cell_bool_r=ramp_cell_bool_r,cell_nums_ramp_r=cell_nums_ramp_r,
                            ramp_cell_bool=ramp_cell_bool,cell_nums_ramp=cell_nums_ramp)
        print(f"{len(cell_nums_ramp)}/{n_neurons} ramping cells")
    else:
        p_result, slope_result, intercept_result, R_result = lin_reg_ramping(delay_resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_all_trials')
        ramp_cell_bool = np.logical_and(p_result<=0.05, np.abs(R_result)>=0.9)
        cell_nums_ramp = np.where(ramp_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_ramp_ident_results_combined.npz'),
                            p_result=p_result, slope_result=slope_result, intercept_result=intercept_result, R_result=R_result,
                            ramp_cell_bool=ramp_cell_bool, cell_nums_ramp=cell_nums_ramp)
        print(f"{len(cell_nums_ramp)}/{n_neurons} ramping cells")
    # breakpoint()

    # # Identifying sequence cells
    if separate_trial_types:
        RB_result_l, z_RB_threshold_result_l = ridge_to_background(left_stim_resp, ramp_cell_bool_l, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_left')
        seq_cell_bool_l = RB_result_l > z_RB_threshold_result_l
        cell_nums_seq_l = np.where(seq_cell_bool_l)[0]  # takes 30 minutes for 256 neurons for 1000 shuffs
        RB_result_r, z_RB_threshold_result_r = ridge_to_background(right_stim_resp, ramp_cell_bool_r, percentile=percentile, n_shuff=n_shuffle,plot=True, save_dir=save_dir, title=f'{seed}_{epi}_right')
        seq_cell_bool_r = RB_result_r > z_RB_threshold_result_r
        cell_nums_seq_r = np.where(seq_cell_bool_r)[0]
        seq_cell_bool = np.logical_or(seq_cell_bool_l, seq_cell_bool_r)
        cell_nums_seq = np.where(seq_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_seq_ident_results_separate.npz'),
                            RB_result_l=RB_result_l, z_RB_threshold_result_l=z_RB_threshold_result_l,seq_cell_bool_l=seq_cell_bool_l, cell_nums_seq_l=cell_nums_seq_l,
                            RB_result_r=RB_result_r, z_RB_threshold_result_r=z_RB_threshold_result_r,seq_cell_bool_r=seq_cell_bool_r,cell_nums_seq_r=cell_nums_seq_r,
                            seq_cell_bool=seq_cell_bool, cell_nums_seq=cell_nums_seq)
        print(f"{len(cell_nums_seq)}/{n_neurons} sequence cells")
    else:
        RB_result, z_RB_threshold_result = ridge_to_background(delay_resp, ramp_cell_bool, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_all_trials')
        seq_cell_bool = RB_result > z_RB_threshold_result  # False if z_RB_threshold_result is nan
        cell_nums_seq = np.where(seq_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir, f'{seed}_{epi}_seq_ident_results_combined.npz'),
                            RB_result=RB_result, z_RB_threshold_result=z_RB_threshold_result,seq_cell_bool=seq_cell_bool,cell_nums_seq=cell_nums_seq)
        print(f"{len(cell_nums_seq)}/{n_neurons} sequence cells")
    # breakpoint()

    if separate_trial_types:
        trial_reliability_score_result_l, trial_reliability_score_threshold_result_l = trial_reliability_score(left_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
        trial_reliable_cell_bool_l = trial_reliability_score_result_l >= trial_reliability_score_threshold_result_l
        trial_reliable_cell_num_l = np.where(trial_reliable_cell_bool_l)[0]
        trial_reliability_score_result_r, trial_reliability_score_threshold_result_r = trial_reliability_score(right_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
        trial_reliable_cell_bool_r = trial_reliability_score_result_r >= trial_reliability_score_threshold_result_r
        trial_reliable_cell_num_r = np.where(trial_reliable_cell_bool_r)[0]
        trial_reliable_cell_bool = np.logical_or(trial_reliable_cell_bool_l, trial_reliable_cell_bool_r)
        trial_reliable_cell_num = np.where(trial_reliable_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_trial_reliability_results_separate.npz'),
                            trial_reliability_score_result_l=trial_reliability_score_result_l, trial_reliability_score_threshold_result_l=trial_reliability_score_threshold_result_l,
                            trial_reliable_cell_bool_l=trial_reliable_cell_bool_l, trial_reliable_cell_num_l=trial_reliable_cell_num_l,
                            trial_reliability_score_result_r=trial_reliability_score_result_r, trial_reliability_score_threshold_result_r=trial_reliability_score_threshold_result_r,
                            trial_reliable_cell_bool_r=trial_reliable_cell_bool_r,trial_reliable_cell_num_r=trial_reliable_cell_num_r,
                            trial_reliable_cell_bool=trial_reliable_cell_bool, trial_reliable_cell_num=trial_reliable_cell_num)
        print(f"{len(trial_reliable_cell_num)}/{n_neurons} trial-reliable cells")
    else:
        trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_score(delay_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)  # Takes 30 minutes for 256 neurons for 1000 shuffs
        trial_reliable_cell_bool = trial_reliability_score_result >= trial_reliability_score_threshold_result
        trial_reliable_cell_num = np.where(trial_reliable_cell_bool)
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_trial_reliability_results_combined.npz'),
                            trial_reliability_score_result=trial_reliability_score_result, trial_reliability_score_threshold_result=trial_reliability_score_threshold_result,
                            trial_reliable_cell_bool=trial_reliable_cell_bool, trial_reliable_cell_num=trial_reliable_cell_num)
        print(f"{len(trial_reliable_cell_num)}/{n_neurons} trial-reliable cells")

    if separate_trial_types:
        I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, n_shuff=n_shuffle, percentile=percentile)  # takes 8 hours to do 1000 shuffles for 256 neurons
        high_temporal_info_cell_bool_l = I_result_l > I_threshold_result_l
        high_temporal_info_cell_nums_l = np.where(high_temporal_info_cell_bool_l)[0]
        I_result_r, I_threshold_result_r = skaggs_temporal_information(right_stim_resp, n_shuff=n_shuffle, percentile=percentile)
        high_temporal_info_cell_bool_r = I_result_r > I_threshold_result_r
        high_temporal_info_cell_nums_r = np.where(high_temporal_info_cell_bool_r)[0]
        high_temporal_info_cell_bool = np.logical_or(high_temporal_info_cell_bool_l, high_temporal_info_cell_bool_r)
        high_temporal_info_cell_nums = np.where(high_temporal_info_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_temporal_info_results_separate.npz'),
                            I_result_l=I_result_l, I_threshold_result_l=I_threshold_result_l,
                            high_temporal_info_cell_bool_l=high_temporal_info_cell_bool_l,high_temporal_info_cell_nums_l=high_temporal_info_cell_nums_l,
                            I_result_r=I_result_r, I_threshold_result_r=I_threshold_result_r,
                            high_temporal_info_cell_bool_r=high_temporal_info_cell_bool_r,high_temporal_info_cell_nums_r=high_temporal_info_cell_nums_r,
                            high_temporal_info_cell_bool=high_temporal_info_cell_bool, high_temporal_info_cell_nums=high_temporal_info_cell_nums)
        print(f"{len(high_temporal_info_cell_nums)}/{n_neurons} high temporal-information cells")
    else:
        I_result, I_threshold_result = skaggs_temporal_information(right_stim_resp, n_shuff=n_shuffle, percentile=percentile)
        high_temporal_info_cell_bool = I_result >= I_threshold_result
        high_temporal_info_cell_num = np.where(high_temporal_info_cell_bool)[0]
        np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_temporal_info_results_combined.npz'),
                            I_result=I_result, I_threshold_result=I_threshold_result,
                            high_temporal_info_cell_bool=high_temporal_info_cell_bool,high_temporal_info_cell_num=high_temporal_info_cell_num)
        print(f"{len(high_temporal_info_cell_num)}/{n_neurons} high temporal-information cells")


cell_nums_nontime = np.delete(np.arange(n_neurons), np.intersect1d(cell_nums_seq, cell_nums_ramp))
# Here, the cell_nums have not been re-arranged according to peak latency yet
delay_resp_ramp = delay_resp[:, :, cell_nums_ramp]
delay_resp_seq = delay_resp[:, :, cell_nums_seq]
delay_resp_nontime = delay_resp[:, :, cell_nums_nontime]
left_stim_resp_ramp = delay_resp_ramp[np.all(stim == [1, 1], axis=1)]
right_stim_resp_ramp = delay_resp_ramp[np.any(stim != [1, 1], axis=1)]
left_stim_resp_seq = delay_resp_seq[np.all(stim == [1, 1], axis=1)]
right_stim_resp_seq = delay_resp_seq[np.any(stim != [1, 1], axis=1)]
left_stim_resp_nontime = delay_resp_nontime[np.all(stim == [1, 1], axis=1)]
right_stim_resp_nontime = delay_resp_nontime[np.any(stim != [1, 1], axis=1)]
n_ramp_neurons = len(cell_nums_ramp)
n_seq_neurons = len(cell_nums_seq)
n_nontime_neurons = len(cell_nums_nontime)

# Re-arrange cell_nums according to peak latency
cell_nums_all, cell_nums_seq, cell_nums_ramp, cell_nums_nontime, \
sorted_matrix_all, sorted_matrix_seq, sorted_matrix_ramp, sorted_matrix_nontime = \
    sort_response_by_peak_latency(delay_resp, cell_nums_ramp, cell_nums_seq, norm=True)

# print('Make a venn diagram of neuron counts...')
make_venn_diagram(cell_nums_ramp, cell_nums_seq, n_neurons, save_dir=save_dir, label=f'{seed}_{epi}', save=True)

print('Sort avg resp analysis...')
plot_sorted_averaged_resp(cell_nums_seq, sorted_matrix_seq, title=env_title+' Sequence cells', remove_nan=True, save_dir=save_dir, save=True)
plot_sorted_averaged_resp(cell_nums_ramp, sorted_matrix_ramp, title=env_title+' Ramping cells', remove_nan=True, save_dir=save_dir, save=True)
plot_sorted_averaged_resp(cell_nums_all, sorted_matrix_all, title=env_title+' All cells', remove_nan=True, save_dir=save_dir, save=True)

print('sort in same order analysis...')
plot_sorted_in_same_order(left_stim_resp_ramp, right_stim_resp_ramp, 'Left', 'Right', big_title=env_title+' Ramping cells', len_delay=len_delay, n_neurons=n_ramp_neurons, save_dir=save_dir, save=True)
plot_sorted_in_same_order(left_stim_resp_seq, right_stim_resp_seq, 'Left', 'Right', big_title=env_title+' Sequence cells', len_delay=len_delay, n_neurons=n_seq_neurons, save_dir=save_dir, save=True)
plot_sorted_in_same_order(left_stim_resp, right_stim_resp, 'Left', 'Right', big_title=env_title+' All cells', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=True)
plot_sorted_in_same_order(correct_resp, incorrect_resp, 'Correct', 'Incorrect', big_title=env_title+' All cells ic', len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=True)

print('decode stim analysis...')
plot_decode_sample_from_single_time(delay_resp, binary_stim, env_title+' All Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=True)
plot_decode_sample_from_single_time(delay_resp_ramp, binary_stim, env_title+' Ramping Cells', n_fold=5, max_iter=100, save_dir=save_dir, save=True)
plot_decode_sample_from_single_time(delay_resp_seq, binary_stim, env_title+' Sequence Cells', n_fold=7, max_iter=100, save_dir=save_dir, save=True)

print('decode time analysis...')
time_decode_lin_reg(delay_resp, len_delay, n_neurons, 1000, title=env_title+' All cells', save_dir=save_dir, save=True)
time_decode_lin_reg(delay_resp_ramp, len_delay, n_ramp_neurons, 1000, title=env_title+' Ramping cells', save_dir=save_dir, save=True)
time_decode_lin_reg(delay_resp_seq, len_delay, n_seq_neurons, 1000, title=env_title+' Sequence cells', save_dir=save_dir, save=True)

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

decode_sample_from_trajectory(delay_loc, stim, save_dir=save_dir, save=True)

print("Mutual info for ramping cells and sequence cells")


ratemap_seq, spatial_occupancy_seq = construct_ratemap(delay_resp_seq, delay_loc)
mutual_info_seq = calculate_mutual_information(ratemap_seq, spatial_occupancy_seq)
shuffled_mutual_info_seq = calculate_shuffled_mutual_information(delay_resp_seq, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_seq, title='seq_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_seq, save_dir=save_dir, save=True)
print("seq: ", np.nanmean(mutual_info_seq))
print("shuffled seq: ", np.nanmean(shuffled_mutual_info_seq))

ratemap_ramp, spatial_occupancy_ramp = construct_ratemap(delay_resp_ramp, delay_loc)
mutual_info_ramp = calculate_mutual_information(ratemap_ramp, spatial_occupancy_ramp)
shuffled_mutual_info_ramp = calculate_shuffled_mutual_information(delay_resp_ramp, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_ramp, title='ramp_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp, save_dir=save_dir, save=True)
print("ramp: ", np.nanmean(mutual_info_ramp))
print("shuffled ramp: ", np.nanmean(shuffled_mutual_info_ramp))

ratemap_nontime, spatial_occupancy_nontime = construct_ratemap(delay_resp_nontime, delay_loc)
mutual_info_nontime = calculate_mutual_information(ratemap_nontime, spatial_occupancy_nontime)
shuffled_mutual_info_nontime = calculate_shuffled_mutual_information(delay_resp_nontime, delay_loc, n_total_episodes)
plot_mutual_info_distribution(mutual_info_ramp, title='non_time_cells', compare=True, shuffled_mutual_info=shuffled_mutual_info_ramp, save_dir=save_dir, save=True)
print("Non-time: ", np.nanmean(mutual_info_nontime))
print("shuffled non-time: ", np.nanmean(shuffled_mutual_info_nontime))


print('Analysis finished')
