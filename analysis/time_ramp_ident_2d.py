"""
Separate time cell and ramping cell identification script to run on cluster.
"""
from cell_identification.time_ramp import *
from analysis_utils import *
#from expts.envs.tunl_1d import *
from linclab_utils import plot_utils
import sys
import argparse


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
env_title = 'Mnemonic_TUNL' if env_type == 'mem' else 'Non-mnemonic_TUNL'
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
    cell_nums_seq_l = np.where(seq_cell_bool_l)[0]
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
    trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_score(delay_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
    trial_reliable_cell_bool = trial_reliability_score_result >= trial_reliability_score_threshold_result
    trial_reliable_cell_num = np.where(trial_reliable_cell_bool)
    np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_trial_reliability_results_combined.npz'),
                        trial_reliability_score_result=trial_reliability_score_result, trial_reliability_score_threshold_result=trial_reliability_score_threshold_result,
                        trial_reliable_cell_bool=trial_reliable_cell_bool, trial_reliable_cell_num=trial_reliable_cell_num)
    print(f"{len(trial_reliable_cell_num)}/{n_neurons} trial-reliable cells")

if separate_trial_types:
    I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, n_shuff=n_shuffle, percentile=percentile)
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

print('Analysis finished')
