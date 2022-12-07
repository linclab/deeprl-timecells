import random
import os
from re import I
from expts.agents.model_1d import *
from expts.envs.int_discrim import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from analysis.cell_identification.time_ramp import separate_ramp_and_seq

# ------------- Control parameters ----------------------------
id = True
load_net = True

# -------------- Define the environment -----------------------
env = IntervalDiscrimination()
env_title = "Interval_Discrimination"

# -------------- Define the agent -----------------------

n_neurons = 512
lr = 1e-5
l2_reg = False
n_total_episodes = 100000
window_size = 100 # for plotting

hidden_types = ['lstm', 'linear']
net_title = 'lstm'
batch_size = 1

# Initializes network
net = AC_Net(
    input_dimensions=2,  # input dim
    action_dimensions=2,  # action dim
    hidden_types=hidden_types,  # hidden types
    hidden_dimensions=[n_neurons, n_neurons], # hidden dims
    batch_size=batch_size)

if load_net:
    load_dir = '2022_02_18_00_00_03_Interval_Discrimination_lstm/net.pt'
    parent_dir = '/Users/ann/Desktop/LiNC_Lab/IntervalDisc/data'
    net.load_state_dict(torch.load(os.path.join(parent_dir, load_dir), map_location=torch.device('cpu')))
    print("Net loaded.")

# Initializes optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# -------------- Define helper functions -----------------------

def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards)+1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode-1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode-1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards


def lesion_experiment(n_total_episodes, lesion_idx):
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
    for i_episode in range(n_total_episodes):
        done = False
        env.reset()
        net.reinit_hid()
        while not done:
            # perform the task
            pol, val = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0), lesion_idx=lesion_idx)  # forward
            if env.task_stage in ['init', 'choice_init']:
                act, p, v = select_action(net, pol, val)
                new_obs, reward, done = env.step(act)
                net.rewards.append(reward)
            else:
                new_obs, reward, done = env.step()

            if env.task_stage == 'choice_init':
                correct_trial[i_episode] = act==env.groundtruth
                #p_loss, v_loss = finish_trial(net, 1, optimizer)
    return np.mean(correct_trial)


def generate_lesion_index(type_lesion, num_lesion, n_neurons, cell_nums_ramp, cell_nums_seq):
    '''
    Arguments:
    - type_lesion: 'random' or 'ramp' or 'seq'. Str.
    - num_lesion: number of cells lesioned. Int.
    Returns:
    - lesion_index
    '''
    if type_lesion == 'random':
        lesion_index = np.random.choice(n_neurons, num_lesion, replace=False)
    elif type_lesion == 'ramp':
        if num_lesion <= len(cell_nums_ramp):
            lesion_index = np.random.choice(cell_nums_ramp, num_lesion, replace=False)
        else:
            lesion_index = np.concatenate((cell_nums_ramp, np.random.choice(cell_nums_seq, num_lesion-len(cell_nums_ramp), replace=False)))
    elif type_lesion == 'seq':
        if num_lesion <= len(cell_nums_seq):
            lesion_index = np.random.choice(cell_nums_seq, num_lesion, replace=False)
        else:
            lesion_index = np.concatenate((cell_nums_seq, np.random.choice(cell_nums_ramp, num_lesion-len(cell_nums_seq), replace=False)))
    return lesion_index

# -------------- Run -----------------------
data = np.load('data/2022_02_18_00_00_03_Interval_Discrimination_lstm/data.npz')  # data.npz file
keep_episode = -10000
stim = data["stim"][keep_episode:,:]
stim1_resp_hx = data["stim1_resp_hx"][keep_episode:,:,:]
stim_set = np.unique(stim)
num_stim = np.max(np.shape(stim_set))

n_lesion = np.arange(13)*10       # from 10 t0 120
postlesion_perf_array = np.zeros((num_stim, 3, len(n_lesion)))
for i_stim, len_stim in enumerate(stim_set):
    resp = stim1_resp_hx[stim[:,0]==len_stim,:,:]
    for i_row, type_lesion in enumerate(['random', 'ramp', 'seq']):
        for i_column, num_lesion in enumerate(n_lesion):
            net.load_state_dict(torch.load(os.path.join(parent_dir, load_dir), map_location=torch.device('cpu')))
            net.eval()
            cell_nums_all, sorted_matrix_all, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp = separate_ramp_and_seq(resp, norm=True)
            lesion_index = generate_lesion_index(type_lesion, num_lesion, n_neurons=n_neurons, cell_nums_ramp=cell_nums_ramp, cell_nums_seq=cell_nums_seq)
            postlesion_perf_array[i_stim, i_row, i_column] = lesion_experiment(n_total_episodes=500, lesion_idx=lesion_index)
            print("Stimulus length:", len_stim, "; Lesion type:", type_lesion, "; Lesion number:", num_lesion, "completed.")

np.savez_compressed('data/lesion.npz', postlesion_perf = postlesion_perf_array)