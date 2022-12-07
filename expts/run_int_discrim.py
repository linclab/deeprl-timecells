import random
import os
from re import I
from agents.model_1d import *
from envs.int_discrim import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from linclab_utils import plot_utils

plot_utils.linclab_plt_defaults()
plot_utils.set_font(font='Helvetica')

# ------------- Control parameters ----------------------------
[ip, it, it_multistimuli, id, ib] = [False, False, False, True, False]
load_net = False
n_total_episodes = 20

# -------------- Define the environment -----------------------

env = IntervalDiscrimination()
env_title = "Interval_Discrimination"


# -------------- Define the agent -----------------------

n_neurons = 512
lr = 1e-5
l2_reg = False
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
    load_dir = '2022_02_27_20_01_03_IntervalTiming_lstm/net.pt'
    parent_dir = '/Users/ann/Desktop/LiNC_Lab/IntervalDisc/data'
    net.load_state_dict(torch.load(os.path.join(parent_dir, load_dir), map_location=torch.device('cpu')))
    print("Net loaded.")

# -------------- Train the agent -----------------------

# Initializes optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Define helper functions
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

def plot_training_rewards(reward_hist):
    avg_reward = bin_rewards(np.sum(reward_hist, axis=1), 1000)
    max_reward = np.ones(n_total_episodes) * 100
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_total_episodes), avg_reward, label='Actual')
    ax.plot(np.arange(n_total_episodes), max_reward, label="Ideal")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Average reward")
    ax.legend(frameon=False)
    ax.set_title("Training performance")
    fig.savefig(path+'/fig.png')

# Train and record
# Initialize arrays for recording

action_hist = np.zeros(n_total_episodes, dtype=np.int8)
correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
stim1_resp_hx = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
stim2_resp_hx = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
delay_resp_hx = np.zeros((n_total_episodes, 20, n_neurons), dtype=np.float32)

for i_episode in range(n_total_episodes):
    if i_episode % 50 == 0:
        print(str(i_episode), "episodes has completed.")
    done = False
    env.reset()
    net.reinit_hid()
    stim[i_episode,0] = env.first_stim
    stim[i_episode,1] = env.second_stim
    while not done:
        # perform the task
        pol, val = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0))  # forward
        if env.task_stage in ['init', 'choice_init']:
            act, p, v = select_action(net, pol, val)
            new_obs, reward, done = env.step(act)
            net.rewards.append(reward)
        else:
            new_obs, reward, done = env.step()

        # record data
        if env.task_stage == 'first_stim' and env.elapsed_t > 0:
            stim1_resp_hx[i_episode, env.elapsed_t-1, :] = net.hx[
                hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
        elif env.task_stage == 'second_stim' and env.elapsed_t > 0:
            stim2_resp_hx[i_episode, env.elapsed_t-1, :] = net.hx[
                hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
        elif env.task_stage == 'delay' and env.elapsed_t > 0:
            delay_resp_hx[i_episode, env.elapsed_t-1, :] = net.hx[
                hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()

        if env.task_stage == 'choice_init':
            action_hist[i_episode] = act
            correct_trial[i_episode] = env.correct_trial

    p_loss, v_loss = finish_trial(net, 1, optimizer)


# Make directory to save data and figures
directory = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{env_title}_{net_title}"
#parent_dir = '/home/mila/z/zixiang.huang/IntervalTiming/data'
parent_dir = '/Users/ann/Desktop/LiNC_Lab/IntervalDisc/data'
path = os.path.join(parent_dir, directory)
os.mkdir(path)


# save data

keep_epi = 50000
np.savez_compressed(path + '/data.npz', action_hist = action_hist[-keep_epi:],
                    stim=stim[-keep_epi:,:], stim1_resp_hx=stim1_resp_hx[-keep_epi:,:,:],
                    stim2_resp_hx=stim2_resp_hx[-keep_epi:,:,:], delay_resp_hx=delay_resp_hx[-keep_epi:,:,:])

# save net
torch.save(net.state_dict(), path+'/net.pt')