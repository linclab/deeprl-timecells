import random
import torch
import numpy as np
from numpy import array
from world1d import *
from model1d import *
import os
from datetime import datetime
import matplotlib.pyplot as plt


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


n_total_episodes = 50
window_size = 2
n_neurons = 512
len_delay = 40
lr = 0.0001

env = TunlEnv(len_delay)
net = AC_Net(3, 4, 1, ['lstm', 'linear'], [n_neurons, n_neurons])
optimizer = torch.optim.Adam(net.parameters(), lr)
env_title = 'Mem'
net_title = 'lstm'

ld = len_delay + 1
delay_resp = np.zeros((n_total_episodes, ld, n_neurons), dtype=np.float32)
stim = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
choice = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
correct_perc = np.zeros(n_total_episodes, dtype=np.float16)

for i_episode in range(n_total_episodes):  # one episode = one sample
    done = False
    env.reset()
    resp = []
    if np.all(env.episode_sample == array([[0, 1, 0]])):  # L
        stim[i_episode] = 0
    elif np.all(env.episode_sample == array([[0, 0, 1]])):  # R
        stim[i_episode] = 1
    net.reinit_hid()
    obs = torch.as_tensor(env.observation)
    while not done:
        pol, val, lin_act = net.forward(obs.float())
        if torch.equal(obs, torch.tensor([[0, 0, 0]])):
            if net.hidden_types[0] == 'linear':
                resp.append(
                    net.cell_out[0].detach().numpy().squeeze())  # pre-relu activity of first layer of linear cell
            elif net.hidden_types[0] == 'lstm':
                resp.append(net.hx[0].clone().detach().numpy().squeeze())  # hidden state of LSTM cell
            elif net.hidden_types[0] == 'gru':
                resp.append(net.hx[0].clone().detach().numpy().squeeze())  # hidden state of GRU cell
        act, p, v = select_action(net, pol, val)
        new_obs, reward, done, info = env.step(act)
        net.rewards.append(reward)
        obs = torch.as_tensor(new_obs)
    choice[i_episode] = act - 1  # 0=L, 1=R
    if stim[i_episode] == choice[i_episode]:
        correct_perc[i_episode] = 1
    delay_resp[i_episode][:len(resp)] = np.asarray(resp)
    p_loss, v_loss = finish_trial(net, 0.99, optimizer)
correct_perc = bin_rewards(correct_perc, window_size=window_size)

# Make directory to save data and figures
#directory = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{env_title}_{net_title}"
#parent_dir = '/home/mila/l/lindongy/tunl1d/data'
#path = os.path.join(parent_dir, directory)
#os.mkdir(path)

fig, ax1 = plt.subplots()
fig.suptitle(env_title)
ax1.plot(np.arange(n_total_episodes), correct_perc, label=net_title)
ax1.set_xlabel('episode')
ax1.set_ylabel('Fraction Nonmatch')
ax1.legend()
plt.show()
#fig.savefig(path+'/fig.png')

#save data
#np.savez_compressed(path + '/data.npz', stim=stim, choice=choice, delay_resp=delay_resp)
#np.savez_compressed(path + '/data_last1000.npz', stim=stim[-1000:], choice=choice[-1000:], delay_resp=delay_resp[-1000:])

# save net
#torch.save(net.state_dict(), path + '/net.pt')