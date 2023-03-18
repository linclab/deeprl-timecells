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
parser.add_argument("--n_shuffle", type=int, default=100, help="number of shuffles to acquire null distribution")
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



# Identify place cells
RB_arr_l, zRB_threshold_arr_l, is_place_cell_l = identify_place_cells(left_stim_resp, left_stim_loc, n_shuff=n_shuffle, percentile=percentile, save_dir=save_dir, title=f'{seed}_{epi}_{percentile}_left')
RB_arr_r, zRB_threshold_arr_r, is_place_cell_r = identify_place_cells(right_stim_resp, right_stim_loc, n_shuff=n_shuffle, percentile=percentile, save_dir=save_dir, title=f'{seed}_{epi}_{percentile}_right')
is_place_cell = np.logical_or(is_place_cell_l, is_place_cell_r)
place_cell_nums = np.where(is_place_cell)[0]
np.savez_compressed(os.path.join(save_dir,f'{seed}_{epi}_{n_shuffle}_{percentile}_place_cell_results.npz'),
                    RB_arr_l=RB_arr_l, zRB_threshold_arr_l=zRB_threshold_arr_l, is_place_cell_l=is_place_cell_l,
                    RB_arr_r=RB_arr_r, zRB_threshold_arr_r=zRB_threshold_arr_r, is_place_cell_r=is_place_cell_r,
                    is_place_cell=is_place_cell, place_cell_nums=place_cell_nums)
print(f"{len(place_cell_nums)}/{n_neurons} place cells")
print('Analysis finished')
