"""
Separate time cell and ramping cell identification script to run on cluster.
"""
from utils_time_ramp import *
from utils_analysis import *
#from expts.envs.tunl_1d import *
import sys
import argparse


parser = argparse.ArgumentParser(description="Non-location-fixed 2D TUNL task simulation")
parser.add_argument("--main_dir",type=str,default='/network/scratch/l/lindongy/timecell/data_collecting/tunl2d',help="main data directory")
parser.add_argument("--data_dir",type=str,default='mem_40_lstm_256_1e-05',help="directory in which .npz is saved")
parser.add_argument("--main_save_dir", type=str, default='/network/scratch/l/lindongy/timecell/data_analysis/tunl2d', help="main directory in which agent-specific directory will be created")
parser.add_argument("--seed", type=int, help="seed to analyse")
parser.add_argument("--episode", type=int, help="ckpt episode to analyse")
parser.add_argument("--normalize", type=bool, default=True, help="normalize each unit's response by its maximum and minimum")
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


# Identifying ramping cells
p_result_l, slope_result_l, intercept_result_l, R_result_l = lin_reg_ramping(left_stim_resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_left')
ramp_cell_bool_l = np.logical_and(p_result_l<=0.05, np.abs(R_result_l)>=0.9)
cell_nums_ramp_l = np.where(ramp_cell_bool_l)[0]
p_result_r, slope_result_r, intercept_result_r, R_result_r = lin_reg_ramping(right_stim_resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_right')
ramp_cell_bool_r = np.logical_and(p_result_r<=0.05, np.abs(R_result_r)>=0.9)
cell_nums_ramp_r = np.where(ramp_cell_bool_r)[0]
ramp_cell_bool = np.logical_or(ramp_cell_bool_l, ramp_cell_bool_r)
cell_nums_ramp = np.where(ramp_cell_bool)[0]
np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_ramp_ident_results.npz'),
                    p_result_l=p_result_l, slope_result_l=slope_result_l, intercept_result_l=intercept_result_l, R_result_l=R_result_l,
                    p_result_r=p_result_r,slope_result_r=slope_result_r, intercept_result_r=intercept_result_r, R_result_r=R_result_r,
                    ramp_cell_bool_l=ramp_cell_bool_l,cell_nums_ramp_l=cell_nums_ramp_l,
                    ramp_cell_bool_r=ramp_cell_bool_r,cell_nums_ramp_r=cell_nums_ramp_r,
                    ramp_cell_bool=ramp_cell_bool,cell_nums_ramp=cell_nums_ramp)
print(f"{len(cell_nums_ramp)}/{n_neurons} ramping cells")

RB_result_l, z_RB_threshold_result_l = ridge_to_background(left_stim_resp, ramp_cell_bool_l, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_left')
seq_cell_bool_l = RB_result_l > z_RB_threshold_result_l
cell_nums_seq_l = np.where(seq_cell_bool_l)[0]
RB_result_r, z_RB_threshold_result_r = ridge_to_background(right_stim_resp, ramp_cell_bool_r, percentile=percentile, n_shuff=n_shuffle,plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_right')
seq_cell_bool_r = RB_result_r > z_RB_threshold_result_r
cell_nums_seq_r = np.where(seq_cell_bool_r)[0]
seq_cell_bool = np.logical_or(seq_cell_bool_l, seq_cell_bool_r)
cell_nums_seq = np.where(seq_cell_bool)[0]
np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_seq_ident_results.npz'),
                    RB_result_l=RB_result_l, z_RB_threshold_result_l=z_RB_threshold_result_l,seq_cell_bool_l=seq_cell_bool_l, cell_nums_seq_l=cell_nums_seq_l,
                    RB_result_r=RB_result_r, z_RB_threshold_result_r=z_RB_threshold_result_r,seq_cell_bool_r=seq_cell_bool_r,cell_nums_seq_r=cell_nums_seq_r,
                    seq_cell_bool=seq_cell_bool, cell_nums_seq=cell_nums_seq)
print(f"{len(cell_nums_seq)}/{n_neurons} significant RB cells")

trial_reliability_score_result_l, trial_reliability_score_threshold_result_l = trial_reliability_vs_shuffle_score(left_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
trial_reliable_cell_bool_l = trial_reliability_score_result_l >= trial_reliability_score_threshold_result_l
trial_reliable_cell_num_l = np.where(trial_reliable_cell_bool_l)[0]
trial_reliability_score_result_r, trial_reliability_score_threshold_result_r = trial_reliability_vs_shuffle_score(right_stim_resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
trial_reliable_cell_bool_r = trial_reliability_score_result_r >= trial_reliability_score_threshold_result_r
trial_reliable_cell_num_r = np.where(trial_reliable_cell_bool_r)[0]
trial_reliable_cell_bool = np.logical_or(trial_reliable_cell_bool_l, trial_reliable_cell_bool_r)
trial_reliable_cell_num = np.where(trial_reliable_cell_bool)[0]
np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_trial_reliability_results.npz'),
                    trial_reliability_score_result_l=trial_reliability_score_result_l, trial_reliability_score_threshold_result_l=trial_reliability_score_threshold_result_l,
                    trial_reliable_cell_bool_l=trial_reliable_cell_bool_l, trial_reliable_cell_num_l=trial_reliable_cell_num_l,
                    trial_reliability_score_result_r=trial_reliability_score_result_r, trial_reliability_score_threshold_result_r=trial_reliability_score_threshold_result_r,
                    trial_reliable_cell_bool_r=trial_reliable_cell_bool_r,trial_reliable_cell_num_r=trial_reliable_cell_num_r,
                    trial_reliable_cell_bool=trial_reliable_cell_bool, trial_reliable_cell_num=trial_reliable_cell_num)
print(f"{len(trial_reliable_cell_num)}/{n_neurons} trial-reliable cells")

# I_result_l, I_threshold_result_l = skaggs_temporal_information(left_stim_resp, n_shuff=n_shuffle, percentile=percentile)
# high_temporal_info_cell_bool_l = I_result_l > I_threshold_result_l
# high_temporal_info_cell_nums_l = np.where(high_temporal_info_cell_bool_l)[0]
# I_result_r, I_threshold_result_r = skaggs_temporal_information(right_stim_resp, n_shuff=n_shuffle, percentile=percentile)
# high_temporal_info_cell_bool_r = I_result_r > I_threshold_result_r
# high_temporal_info_cell_nums_r = np.where(high_temporal_info_cell_bool_r)[0]
# high_temporal_info_cell_bool = np.logical_or(high_temporal_info_cell_bool_l, high_temporal_info_cell_bool_r)
# high_temporal_info_cell_nums = np.where(high_temporal_info_cell_bool)[0]
# np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_temporal_info_results.npz'),
#                     I_result_l=I_result_l, I_threshold_result_l=I_threshold_result_l,
#                     high_temporal_info_cell_bool_l=high_temporal_info_cell_bool_l,high_temporal_info_cell_nums_l=high_temporal_info_cell_nums_l,
#                     I_result_r=I_result_r, I_threshold_result_r=I_threshold_result_r,
#                     high_temporal_info_cell_bool_r=high_temporal_info_cell_bool_r,high_temporal_info_cell_nums_r=high_temporal_info_cell_nums_r,
#                     high_temporal_info_cell_bool=high_temporal_info_cell_bool, high_temporal_info_cell_nums=high_temporal_info_cell_nums)
# print(f"{len(high_temporal_info_cell_nums)}/{n_neurons} high temporal-information cells")

# Identify time cells: combination of RB and trial reliability
time_cell_nums_l = np.intersect1d(cell_nums_seq_l, trial_reliable_cell_num_l)
time_cell_nums_r = np.intersect1d(cell_nums_seq_r, trial_reliable_cell_num_r)
time_cell_nums = np.intersect1d(time_cell_nums_l, time_cell_nums_r)
np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_time_cell_results.npz'),
                    time_cell_nums_l=time_cell_nums_l, time_cell_nums_r=time_cell_nums_r, time_cell_nums=time_cell_nums)
print(f"{len(time_cell_nums)}/{n_neurons} time cells")


