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
    os.makedirs(save_dir)
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
    reshape_resp = np.reshape(stim1_resp, (n_total_episodes*40, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    stim1_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(stim2_resp, (n_total_episodes*40, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    stim2_resp = np.reshape(reshape_resp, (n_total_episodes, 40, n_neurons))

    reshape_resp = np.reshape(delay_resp, (n_total_episodes*20, n_neurons))
    reshape_resp = (reshape_resp - np.min(reshape_resp, axis=0, keepdims=True)) / np.ptp(reshape_resp, axis=0, keepdims=True)
    delay_resp = np.reshape(reshape_resp, (n_total_episodes, 20, n_neurons))

for (resp, stimulus, label) in zip([stim1_resp,stim2_resp, delay_resp], [stim[:,0],stim[:,1], None], ['stimulus_1', 'stimulus_2', 'delay']):
    print(f"analysing data from {label}")
