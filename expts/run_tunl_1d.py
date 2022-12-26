import random
import torch
import numpy as np
from numpy import array
from envs.tunl_1d import TunlEnv, TunlEnv_nomem
from agents.model_1d import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards) + 1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode - 1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode - 1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards

parser = argparse.ArgumentParser(description="Head-fixed 1D TUNL task simulation")
parser.add_argument("--n_total_episodes",type=int,default=50000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=5000,help="Save model every this number of episodes")
parser.add_argument("--record_data", type=bool, default=False, help="Whether to collect data while training.")
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl1d")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--len_delay", type=int, default=40, help="Number of timesteps in the delay period")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
parser.add_argument("--env_type", type=str, default='mem', help="type of environment: mem or nomem")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)

n_total_episodes = argsdict['n_total_episodes']
save_ckpt_per_episodes = argsdict['save_ckpt_per_episodes']
save_ckpts = True if argsdict['save_ckpts'] == True or argsdict['save_ckpts'] == 'True' else False
record_data = True if argsdict['record_data'] == True or argsdict['record_data'] == 'True' else False
load_model_path = argsdict['load_model_path']
window_size = n_total_episodes // 10
n_neurons = argsdict["n_neurons"]
len_delay = argsdict['len_delay']
lr = argsdict['lr']
env_type = argsdict['env_type']
hidden_type = argsdict['hidden_type']

# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl1d'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/tunl1d'
save_dir = os.path.join(main_dir, f'{env_type}_{len_delay}_{hidden_type}_{n_neurons}_{lr}')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(f'Saved tp {save_dir}')

# Setting up cuda and seeds
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")  # Not using cuda for TUNL 1d
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])

env = TunlEnv(len_delay) if env_type=='mem' else TunlEnv_nomem(len_delay)
net = AC_Net(3, 4, 1, ['linear', hidden_type, 'linear'], [n_neurons, n_neurons, n_neurons])
optimizer = torch.optim.Adam(net.parameters(), lr)
env_title = 'Mnemonic' if env_type == 'mem' else 'Non-mnemonic'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'
ld = len_delay + 1


# Load existing model
if load_model_path=='None':
    ckpt_name = 'untrained_agent'  # placeholder ckptname in case we want to save data in the end
else:
    ckpt_name = load_model_path.replace('/', '_')
    # assert loaded model has congruent hidden type and n_neurons
    assert hidden_type in ckpt_name, 'Must load network with the same hidden type'
    assert str(n_neurons) in ckpt_name, 'Must load network with the same number of hidden neurons'
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl1d', load_model_path)))

stim = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
choice = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
correct_perc = np.zeros(n_total_episodes, dtype=np.float16)
if record_data:
    delay_resp = np.zeros((n_total_episodes, ld, n_neurons), dtype=np.float32)

for i_episode in tqdm(range(n_total_episodes)):  # one episode = one sample
    done = False
    env.reset()
    if np.all(env.episode_sample == array([[0, 1, 0]])):  # L
        stim[i_episode] = 0
    elif np.all(env.episode_sample == array([[0, 0, 1]])):  # R
        stim[i_episode] = 1
    if record_data:
        resp = []
    net.reinit_hid()
    obs = torch.as_tensor(env.observation)
    while not done:
        pol, val, lin_act = net.forward(obs.float())
        if record_data and torch.equal(obs, torch.tensor([[0, 0, 0]])):
            if net.hidden_types[1] == 'linear':
                resp.append(
                    net.cell_out[1].detach().numpy().squeeze())  # pre-relu activity of first layer of linear cell
            elif net.hidden_types[1] == 'lstm':
                resp.append(net.hx[1].clone().detach().numpy().squeeze())  # hidden state of LSTM cell
            elif net.hidden_types[1] == 'gru':
                resp.append(net.hx[1].clone().detach().numpy().squeeze())  # hidden state of GRU cell
        act, p, v = select_action(net, pol, val)
        new_obs, reward, done, info = env.step(act)
        net.rewards.append(reward)
        obs = torch.as_tensor(new_obs)

    choice[i_episode] = act - 1  # 0=L, 1=R
    if stim[i_episode] + choice[i_episode] == 1:  # nonmatch
        correct_perc[i_episode] = 1
    if record_data:
        delay_resp[i_episode][:len(resp)] = np.asarray(resp)
    p_loss, v_loss = finish_trial(net, 0.99, optimizer)
    if (i_episode+1) % save_ckpt_per_episodes == 0:
        print(f'Episode {i_episode}, {np.mean(correct_perc[:i_episode+1])*100:.3f}% correct')
        if save_ckpts:
            torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')
binned_correct_perc = bin_rewards(correct_perc, window_size=window_size)


fig, ax1 = plt.subplots()
fig.suptitle(env_title)
ax1.plot(np.arange(n_total_episodes), binned_correct_perc, label=net_title)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Fraction Nonmatch')
ax1.set_ylim(0,1)
ax1.set_title(f'{env_title} TUNL')
ax1.legend()
#plt.show()
fig.savefig(save_dir + f'/total_{n_total_episodes}episodes_performance.svg')

# save data
if record_data:
    np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, choice=choice, delay_resp=delay_resp)
else:
    np.savez_compressed(save_dir + f'/total_{n_total_episodes}episodes_performance_data.npz', stim=stim, choice=choice)

