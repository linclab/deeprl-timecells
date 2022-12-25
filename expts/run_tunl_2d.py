import random
import os
from envs.tunl_2d import *
from agents.model_2d import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

mem, nomem, mem_vd, nomem_vd = [False, False, False, False]
mem = True
env_title = 'Tunl Mem'

if mem or nomem:
    ld = 40
elif mem_vd or nomem_vd:
    len_delays = [20, 40, 60]
    len_delays_p = [1, 1, 1]
    ld = max(len_delays)

len_edge = 7
rwd = 100
inc_rwd = -20
step_rwd = -0.1
poke_rwd = 5
rng_seed = 1234

if mem:
    env = Tunl(ld, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, rng_seed)
elif nomem:
    env = Tunl_nomem(ld, len_edge, rwd, step_rwd, poke_rwd, rng_seed)
elif mem_vd:
    env = Tunl_vd(len_delays, len_delays_p, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, rng_seed)
elif nomem_vd:
    env = Tunl_nomem_vd(len_delays, len_delays_p, len_edge, rwd, step_rwd, poke_rwd, rng_seed)

n_neurons = 512
lr = 1e-5
batch_size = 1
rfsize = 2
padding = 0
stride = 1
dilation = 1
conv_1_features = 16
conv_2_features = 32
hidden_types = ['conv', 'pool', 'conv', 'pool', 'lstm', 'linear']
net_title = hidden_types[4]
l2_reg = False

n_total_episodes = 50000
window_size = 5000  # for plotting

# Define conv & pool layer sizes
layer_1_out_h, layer_1_out_w = conv_output(env.h, env.w, padding, dilation, rfsize, stride)
layer_2_out_h, layer_2_out_w = conv_output(layer_1_out_h, layer_1_out_w, padding, dilation, rfsize, stride)
layer_3_out_h, layer_3_out_w = conv_output(layer_2_out_h, layer_2_out_w, padding, dilation, rfsize, stride)
layer_4_out_h, layer_4_out_w = conv_output(layer_3_out_h, layer_3_out_w, padding, dilation, rfsize, stride)

# Initializes network

net = AC_Net(
    input_dimensions=(env.h, env.w, 3),  # input dim
    action_dimensions=6,  # action dim
    hidden_types=hidden_types,  # hidden types
    hidden_dimensions=[
        (layer_1_out_h, layer_1_out_w, conv_1_features),  # conv
        (layer_2_out_h, layer_2_out_w, conv_1_features),  # pool
        (layer_3_out_h, layer_3_out_w, conv_2_features),  # conv
        (layer_4_out_h, layer_4_out_w, conv_2_features),  # pool
        n_neurons,
        n_neurons],  # hidden_dims
    batch_size=batch_size,
    rfsize=rfsize,
    padding=padding,
    stride=stride)

# If load pre-trained network
'''
load_dir = '2021_03_07_21_58_43_7_10_1e-05/net.pt'
parent_dir = '/home/mila/l/lindongy/tunl2d/data'
net.load_state_dict(torch.load(os.path.join(parent_dir, load_dir)))
'''

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


def ideal_nav_rwd(env, len_edge, len_delay, step_rwd, poke_rwd):
    """
    Given env, len_edge, len_delay, step_rwd, poke_rwd, return the ideal navigation reward for a single episode.
    Use after env.reset().
    """
    ideal_nav_reward = (env.dist_to_init + 3 * (len_edge - 1) - min(
        (len_delay, len_edge - 1))) * step_rwd + 3 * poke_rwd
    return ideal_nav_reward


# Train and record
# Initialize arrays for recording
if mem_vd or nomem_vd:
    len_delay = np.zeros(n_total_episodes, dtype=np.int8)  # length of delay for each trial

if mem or mem_vd:
    ct = np.zeros(n_total_episodes, dtype=np.int8)  # whether it's a correction trial or not

stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
correct_perc = np.zeros(n_total_episodes, dtype=np.float16)
choice = np.zeros((n_total_episodes, 2), dtype=np.int8)  # record the location when done
delay_loc = np.zeros((n_total_episodes, ld, 2), dtype=np.int16)  # location during delay
delay_resp_hx = np.zeros((n_total_episodes, ld, n_neurons), dtype=np.float32)  # hidden states during delay
delay_resp_cx = np.zeros((n_total_episodes, ld, n_neurons), dtype=np.float32)  # cell states during delay
ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)


for i_episode in range(n_total_episodes):
    done = False
    env.reset()
    ideal_nav_rwds[i_episode] = ideal_nav_rwd(env, len_edge, env.len_delay, step_rwd, poke_rwd)
    net.reinit_hid()
    stim[i_episode] = env.sample_loc
    if mem or mem_vd:
        ct[i_episode] = int(env.correction_trial)
    if mem_vd or nomem_vd:
        len_delay[i_episode] = env.len_delay  # For vd or it only
    while not done:
        pol, val = net.forward(
            torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float()
        )  # forward
        if env.indelay:  # record location and neural responses
            delay_loc[i_episode, env.delay_t - 1, :] = np.asarray(env.current_loc)
            delay_resp_hx[i_episode, env.delay_t - 1, :] = net.hx[
                hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            delay_resp_cx[i_episode, env.delay_t - 1, :] = net.cx[
                hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
        act, p, v = select_action(net, pol, val)
        new_obs, reward, done, info = env.step(act)
        net.rewards.append(reward)
    choice[i_episode] = env.current_loc
    if env.reward == rwd:
        correct_perc[i_episode] = 1
    epi_nav_reward[i_episode] = env.nav_reward
    p_loss, v_loss = finish_trial(net, 0.99, optimizer)

avg_nav_rewards = bin_rewards(epi_nav_reward, window_size)
correct_perc = bin_rewards(correct_perc, window_size)

# Make directory to save data and figures
directory = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{env_title}_{net_title}"
parent_dir = '/home/mila/l/lindongy/tunl2d/data'
path = os.path.join(parent_dir, directory)
os.mkdir(path)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))

fig.suptitle(env_title)

ax1.plot(np.arange(n_total_episodes), avg_nav_rewards, label=net_title)
ax1.plot(np.arange(n_total_episodes), ideal_nav_rwds, label="Ideal navig reward")
ax1.set_xlabel('episode')
ax1.set_ylabel('navigation reward')
ax1.legend()

ax2.plot(np.arange(n_total_episodes), correct_perc, label=net_title)
ax2.set_xlabel('episode')
ax2.set_ylabel('correct %')
ax2.legend()


# plt.show()
fig.savefig(path+'/fig.png')

# save data
if mem:
    np.savez_compressed(path + '/data.npz', stim=stim, choice=choice, ct=ct, delay_loc=delay_loc,
                        delay_resp_hx=delay_resp_hx,
                        delay_resp_cx=delay_resp_cx)
elif nomem:
    np.savez_compressed(path + '/data.npz', stim=stim, choice=choice, delay_loc=delay_loc, delay_resp_hx=delay_resp_hx,
                        delay_resp_cx=delay_resp_cx)
elif mem_vd:
    np.savez_compressed(path + '/data.npz', stim=stim, choice=choice, ct=ct, len_delay=len_delay, delay_loc=delay_loc,
                        delay_resp_hx=delay_resp_hx, delay_resp_cx=delay_resp_cx)
elif nomem_vd:
    np.savez_compressed(path + '/data.npz', stim=stim, choice=choice, len_delay=len_delay, delay_loc=delay_loc,
                        delay_resp_hx=delay_resp_hx, delay_resp_cx=delay_resp_cx)

# save net
torch.save(net.state_dict(), path+'/net.pt')
