import random
import os
from envs.tunl_2d import Tunl_incentive, Tunl_nomem_incentive
from agents.model_2d import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import re
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

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


parser = argparse.ArgumentParser(description="Non-location-fixed TUNL 2D task simulation")
parser.add_argument("--n_total_episodes",type=int,default=50000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=5000,help="Save model every this number of episodes")
parser.add_argument("--record_data", type=bool, default=False, help="Whether to collect data while training. If False, don't pass anything. If true, pass True.")
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl2d")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes. If False, don't pass anything. If true, pass True.")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--len_delay", type=int, default=40, help="Number of timesteps in the delay period")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
parser.add_argument("--env_type", type=str, default='mem', help="type of environment: mem or nomem")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--len_edge", type=int, default=7, help="Length of the longer edge of the triangular arena. Must be odd number >=5")
parser.add_argument("--nonmatch_reward", type=int, default=100, help="Magnitude of reward when agent chooses nonmatch side")
parser.add_argument("--incorrect_reward", type=int, default=-20, help="Magnitude of reward when agent chooses match side")
parser.add_argument("--step_reward", type=float, default=-0.1, help="Magnitude of reward for each action agent takes, to ensure shortest path")
parser.add_argument("--poke_reward", type=int, default=5, help="Magnitude of reward when agent pokes signal to proceed")
parser.add_argument("--save_performance_fig", type=bool, default=False, help="If False, don't pass anything. If true, pass True.")
parser.add_argument("--p_dropout", type=float, default=0.0, help="dropout probability")
parser.add_argument("--dropout_type", type=int, default=None, help="location of dropout (could be 1,2,3,or 4)")
parser.add_argument("--p_small_reward", type=float, default=0.0, help="probability of small reward")
parser.add_argument("--a_small_reward", type=int, default=3, help="magnitude of small reward, which is 1/a of nonmatch reward")
args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)

n_total_episodes = argsdict['n_total_episodes']
save_ckpt_per_episodes = argsdict['save_ckpt_per_episodes']
save_ckpts = True if argsdict['save_ckpts'] == True or argsdict['save_ckpts'] == 'True' else False
record_data = True if argsdict['record_data'] == True or argsdict['record_data'] == 'True' else False
save_performance_fig = True if argsdict['save_performance_fig'] == True or argsdict['save_performance_fig'] == 'True' else False
load_model_path = argsdict['load_model_path']
window_size = n_total_episodes // 10
n_neurons = argsdict["n_neurons"]
len_delay = argsdict['len_delay']
lr = argsdict['lr']
env_type = argsdict['env_type']
hidden_type = argsdict['hidden_type']
len_edge = argsdict['len_edge']
rwd = argsdict['nonmatch_reward']
inc_rwd = argsdict['incorrect_reward']
step_rwd = argsdict['step_reward']
poke_rwd = argsdict['poke_reward']
seed = argsdict['seed']
weight_decay = argsdict['weight_decay']
p_dropout = argsdict['p_dropout']
dropout_type = argsdict['dropout_type']
p_small_reward = argsdict['p_small_reward']
a_small_reward = argsdict['a_small_reward']

# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl2d'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/tunl2d'
save_dir_str = f'{env_type}_{len_delay}_{hidden_type}_{n_neurons}_{lr}_smallrwd{p_small_reward}_{a_small_reward}'
if weight_decay != 0:
    save_dir_str += f'_wd{weight_decay}'
if p_dropout != 0:
    save_dir_str += f'_p{p_dropout}_{dropout_type}'
save_dir = os.path.join(main_dir, save_dir_str)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(f'Saved to {save_dir}')

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])


#env = Tunl(len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, seed) if env_type=='mem' else Tunl_nomem(len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, seed)
env = Tunl_incentive(len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, p_small_reward, a_small_reward, rng_seed=seed) if env_type=='mem' else Tunl_nomem_incentive(len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, p_small_reward, a_small_reward, rng_seed=seed)

rfsize = 2
padding = 0
stride = 1
dilation = 1
conv_1_features = 16
conv_2_features = 32

# Define conv & pool layer sizes
layer_1_out_h, layer_1_out_w = conv_output(env.h, env.w, padding, dilation, rfsize, stride)
layer_2_out_h, layer_2_out_w = conv_output(layer_1_out_h, layer_1_out_w, padding, dilation, rfsize, stride)
layer_3_out_h, layer_3_out_w = conv_output(layer_2_out_h, layer_2_out_w, padding, dilation, rfsize, stride)
layer_4_out_h, layer_4_out_w = conv_output(layer_3_out_h, layer_3_out_w, padding, dilation, rfsize, stride)

# Initializes network

net = AC_Net(
    input_dimensions=(env.h, env.w, 3),  # input dim
    action_dimensions=6,  # action dim
    hidden_types=['conv', 'pool', 'conv', 'pool', hidden_type, 'linear'],  # hidden types
    hidden_dimensions=[
        (layer_1_out_h, layer_1_out_w, conv_1_features),  # conv
        (layer_2_out_h, layer_2_out_w, conv_1_features),  # pool
        (layer_3_out_h, layer_3_out_w, conv_2_features),  # conv
        (layer_4_out_h, layer_4_out_w, conv_2_features),  # pool
        n_neurons,
        n_neurons],  # hidden_dims
    p_dropout=p_dropout,
    dropout_type=dropout_type,
    batch_size=1,
    rfsize=rfsize,
    padding=padding,
    stride=stride)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
env_title = 'Mnemonic' if env_type == 'mem' else 'Non-mnemonic'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

# Load existing model
if load_model_path=='None':
    ckpt_name = f'seed_{seed}_untrained_agent_weight_frozen'  # placeholder ckptname in case we want to save data in the end
else:
    ckpt_name = load_model_path.replace('/', '_')
    pt_name = load_model_path.split('/')[1]  # seed_3_epi199999.pt
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    loaded_ckpt_seed = int(pt[1])
    loaded_ckpt_episode = int(pt[2])
    # assert loaded model has congruent hidden type and n_neurons
    assert hidden_type in ckpt_name, 'Must load network with the same hidden type'
    assert str(n_neurons) in ckpt_name, 'Must load network with the same number of hidden neurons'
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl2d', load_model_path)))


# Initialize arrays for recording
if env_type == 'mem':
    ct = np.zeros(n_total_episodes, dtype=np.int8)  # whether it's a correction trial or not

stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
nonmatch_perc = np.zeros(n_total_episodes, dtype=np.float16)
choice = np.zeros((n_total_episodes, 2), dtype=np.int8)  # record the location when done
ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)
nomem_perf = np.zeros(n_total_episodes, dtype=np.float16)
if record_data:  # list of lists. Each sublist is data from one episode
    neural_activity = []
    action = []
    location = []
    rwd = []
epi_small_reward = np.zeros(n_total_episodes, dtype=np.float16)
epi_to_incentivize = np.zeros(n_total_episodes, dtype=np.float16)

for i_episode in tqdm(range(n_total_episodes)):
    done = False
    env.reset()
    ideal_nav_rwds[i_episode] = ideal_nav_rwd(env, len_edge, env.len_delay, step_rwd, poke_rwd)
    epi_to_incentivize[i_episode] = env.to_incentivize
    net.reinit_hid()
    stim[i_episode] = env.sample_loc
    if env_type=='mem':
        ct[i_episode] = int(env.correction_trial)
    if record_data:
        neural_activity.append([])
        action.append([])
        location.append([])
        rwd.append([])
    small_reward = []
    while not done:
        obs = torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device)
        pol, val = net.forward(obs)  # forward
        act, p, v = select_action(net, pol, val)
        if record_data:
            location[i_episode].append(env.current_loc)
            neural_activity[i_episode].append(net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())
            action[i_episode].append(act)
        new_obs, reward, done, info = env.step(act)
        if record_data:
            rwd[i_episode].append(reward)
        if env.indelay:
            small_reward.append(env.reward)
        net.rewards.append(reward)
    epi_small_reward[i_episode] = np.sum(small_reward[1:])
    choice[i_episode] = env.current_loc
    if np.any(stim[i_episode] != choice[i_episode]):  # nonmatch
        nonmatch_perc[i_episode] = 1  # check
    nomem_perf[i_episode] = 1 if reward == rwd else 0  # Touch L to get reward
    epi_nav_reward[i_episode] = env.nav_reward
    if record_data:
        del net.rewards[:]
        del net.saved_actions[:]
    else:
        p_loss, v_loss = finish_trial(net, 0.99, optimizer)
    if (i_episode+1) % save_ckpt_per_episodes == 0:
        if load_model_path != 'None':
            # report epi_small_reward
            print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(epi_small_reward[i_episode+1-save_ckpt_per_episodes:i_episode+1])} avg small reward in the last {save_ckpt_per_episodes} episodes, avg {np.mean(epi_small_reward[:i_episode+1])} avg small reward')
            if env_type == 'mem':
                print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(nonmatch_perc[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% nonmatch in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nonmatch_perc[:i_episode+1])*100:.3f}% nonmatch')
            else:
                print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(nomem_perf[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nomem_perf[:i_episode+1])*100:.3f}% correct')
        else:
            # report epi_small_reward
            print(f'Episode {i_episode}, {np.mean(epi_small_reward[i_episode+1-save_ckpt_per_episodes:i_episode+1])} avg small reward in the last {save_ckpt_per_episodes} episodes, avg {np.mean(epi_small_reward[:i_episode+1])} avg small reward')
            if env_type == 'mem':
                print(f'Episode {i_episode}, {np.mean(nonmatch_perc[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% nonmatch in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nonmatch_perc[:i_episode+1])*100:.3f}% nonmatch')
            else:
                print(f'Episode {i_episode}, {np.mean(nomem_perf[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nomem_perf[:i_episode+1])*100:.3f}% correct')
        if save_ckpts:
            if load_model_path != 'None':
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode+loaded_ckpt_episode}.pt')
            else:
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')
avg_nav_rewards = bin_rewards(epi_nav_reward, window_size)
binned_nonmatch_perc = bin_rewards(nonmatch_perc, window_size)
binned_nomem_perf = bin_rewards(nomem_perf, window_size)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(6, 6))
fig.suptitle(f'{env_title} TUNL')
ax1.plot(np.arange(n_total_episodes), avg_nav_rewards, label=net_title)
ax1.plot(np.arange(n_total_episodes), ideal_nav_rwds, label="Ideal navigation reward")
ax1.set_xlabel('Episode')
ax1.set_ylabel('Navigation reward')
ax1.legend()

# ax2.plot(np.arange(n_total_episodes), binned_nonmatch_perc, label=net_title)
ax2.plot(np.arange(n_total_episodes), binned_nomem_perf, label=net_title)
ax2.set_xlabel('Episode')
# ax2.set_ylabel('Fraction Nonmatch')
ax2.set_ylabel('% correct')
ax2.set_ylim(0,1)
ax2.legend()
if save_performance_fig:
    fig.savefig(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance.svg')

# save data
if record_data:
    if env_type=='mem':
        np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, choice=choice, ct=ct,
                            neural_activity=neural_activity,
                            action=action,
                            location=location,
                            reward=rwd,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds,
                            epi_small_reward=epi_small_reward,
                            epi_to_incentivize=epi_to_incentivize)
    else:
        np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, choice=choice,
                            neural_activity=neural_activity,
                            action=action,
                            location=location,
                            reward=rwd,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds,
                            epi_small_reward=epi_small_reward,
                            epi_to_incentivize=epi_to_incentivize)
else:
    if env_type=='mem':
        np.savez_compressed(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance_data.npz', stim=stim, choice=choice,
                            ct=ct,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds,
                            epi_small_reward=epi_small_reward,
                            epi_to_incentivize=epi_to_incentivize)
    else:
        np.savez_compressed(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance_data.npz', stim=stim, choice=choice,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds,
                            epi_small_reward=epi_small_reward,
                            epi_to_incentivize=epi_to_incentivize)
