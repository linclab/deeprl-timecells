from utils_time_ramp import *
from utils_analysis import *
import sys
import argparse


parser = argparse.ArgumentParser(description="Non location-fixed 2D interval discrimination task simulation")
parser.add_argument("--main_dir",type=str,default='/network/scratch/l/lindongy/timecell/data_collecting/timing2d',help="main data directory")
parser.add_argument("--data_dir",type=str,default='lstm_256_5e-06',help="directory in which .npz is saved")
parser.add_argument("--main_save_dir", type=str, default='/network/scratch/l/lindongy/timecell/data_analysis/timing2d', help="main directory in which agent-specific directory will be created")
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
    os.makedirs(save_dir, exist_ok=True)
seed = argsdict['seed']
epi = argsdict['episode']
n_shuffle = argsdict['n_shuffle']
percentile = argsdict['percentile']
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
env_title = 'Interval_Discrimination_2D'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

reward_hist = data["reward_hist"]
correct_perc = data["correct_perc"]
stim = data["stim"]
stim1_resp = data["stim_1_resp"]  # Note: this could also be linear activity of Feedforward network
stim2_resp = data["stim_2_resp"]  # Note: this could also be linear activity of Feedforward network
delay_resp = data["delay_resp_hx"]  # Note: this could also be linear activity of Feedforward network
n_total_episodes = np.shape(stim)[0]

normalize = True if argsdict['normalize'] == True or argsdict['normalize'] == 'True' else False
if normalize:
    # normalize each unit's response by its maximum and minimum, ignoring the 0s beyond stimulus length as indicated in stim
    for i_neuron in range(n_neurons):
        # swap 0's in stim1_resp and stim2_resp with nan
        stim1_resp[:, :, i_neuron][stim1_resp[:, :, i_neuron] == 0] = np.nan
        stim2_resp[:, :, i_neuron][stim2_resp[:, :, i_neuron] == 0] = np.nan
        # normalize across stim1_resp, stim2_resp, and delay_resp
        min_act = np.nanmin(np.concatenate((stim1_resp[:, :, i_neuron], stim2_resp[:, :, i_neuron], delay_resp[:, :, i_neuron]), axis=1))
        max_act = np.nanmax(np.concatenate((stim1_resp[:, :, i_neuron], stim2_resp[:, :, i_neuron], delay_resp[:, :, i_neuron]), axis=1))
        stim1_resp[:, :, i_neuron] = (stim1_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        stim2_resp[:, :, i_neuron] = (stim2_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        delay_resp[:, :, i_neuron] = (delay_resp[:, :, i_neuron] - min_act) / (max_act - min_act)
        # swap nan's back to 0's
        stim1_resp[:, :, i_neuron][np.isnan(stim1_resp[:, :, i_neuron])] = 0
        stim2_resp[:, :, i_neuron][np.isnan(stim2_resp[:, :, i_neuron])] = 0
        delay_resp[:, :, i_neuron][np.isnan(delay_resp[:, :, i_neuron])] = 0

time_cell_id = []
ramp_cell_id = []
# Define period that we want to analyse
for (resp, stimulus, label) in zip([stim1_resp,stim2_resp, delay_resp], [stim[:,0],stim[:,1], None], ['stimulus_1', 'stimulus_2', 'delay']):
    print(f"analysing data from {label}")
    # Identifying ramping cells (Toso 2021)
    if label=='delay':
        p_result, slope_result, intercept_result, R_result = lin_reg_ramping(resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}')
    else:
        p_result, slope_result, intercept_result, R_result = lin_reg_ramping_varying_duration(resp, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}')
    ramp_cell_bool = np.logical_and(p_result<=0.05, np.abs(R_result)>=0.9)
    cell_nums_ramp = np.where(ramp_cell_bool)[0]
    np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_ramp_ident_results.npz'),
                        p_result=p_result, slope_result=slope_result, intercept_result=intercept_result, R_result=R_result,
                        ramp_cell_bool=ramp_cell_bool, cell_nums_ramp=cell_nums_ramp)
    print(f"{len(cell_nums_ramp)}/{n_neurons} ramping cells")
    ramp_cell_id.append(cell_nums_ramp)

    # # Identifying sequence cells
    if label == 'delay':
        RB_result, z_RB_threshold_result = ridge_to_background(resp, ramp_cell_bool, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}')
    else:
        RB_result, z_RB_threshold_result = ridge_to_background_varying_duration(resp, stim,  ramp_cell_bool, percentile=percentile, n_shuff=n_shuffle, plot=True, save_dir=save_dir, title=f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}')
    seq_cell_bool = RB_result > z_RB_threshold_result
    cell_nums_seq = np.where(seq_cell_bool)[0]
    np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_seq_ident_results.npz'),
                        RB_result=RB_result, z_RB_threshold_result=z_RB_threshold_result,seq_cell_bool=seq_cell_bool, cell_nums_seq=cell_nums_seq,)
    print(f"{len(cell_nums_seq)}/{n_neurons} significant RB cells")

    if label=='delay':
        trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_vs_shuffle_score(resp, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
    else:
        trial_reliability_score_result, trial_reliability_score_threshold_result = trial_reliability_vs_shuffle_score_varying_duration(resp, stim, split='odd-even', percentile=percentile, n_shuff=n_shuffle)
    trial_reliable_cell_bool = trial_reliability_score_result >= trial_reliability_score_threshold_result
    trial_reliable_cell_num = np.where(trial_reliable_cell_bool)[0]
    np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_trial_reliability_results.npz'),
                        trial_reliability_score_result=trial_reliability_score_result, trial_reliability_score_threshold_result=trial_reliability_score_threshold_result,
                        trial_reliable_cell_bool=trial_reliable_cell_bool, trial_reliable_cell_num=trial_reliable_cell_num)
    print(f"{len(trial_reliable_cell_num)}/{n_neurons} trial-reliable cells")

    # Identify time cells: combination of RB and trial reliability
    time_cell_nums = np.intersect1d(cell_nums_seq, trial_reliable_cell_num)
    np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_{label}_time_cell_results.npz'),time_cell_nums=time_cell_nums)
    print(f"{len(time_cell_nums)}/{n_neurons} time cells")

    time_cell_id.append(time_cell_nums)

print(f"Time cell ID: \nStim1:{time_cell_id[0]}\nStim2:{time_cell_id[1]}\nDelay:{time_cell_id[2]}")
print(f"Ramping cell ID: \nStim1:{ramp_cell_id[0]}\nStim2:{ramp_cell_id[1]}\nDelay:{ramp_cell_id[2]}")
print('Analysis finished')