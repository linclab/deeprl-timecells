import random
import os
from agents.model_2d import *
from envs.int_discrim_2d import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import  argparse
from tqdm import tqdm
# from analysis.linclab_utils import plot_utils
#
# plot_utils.linclab_plt_defaults()
# plot_utils.set_font(font='Helvetica')


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


def ideal_nav_rwd(env, step_rwd, poke_rwd, rwd):
    """
    Given env, len_edge, len_delay, step_rwd, poke_rwd, return the ideal navigation reward for a single episode.
    Use after env.reset().
    """
    ideal_nav_reward = env.dist_to_init * step_rwd + poke_rwd + rwd
    return ideal_nav_reward


parser = argparse.ArgumentParser(description="Non head-fixed Interval Discrimination task simulation")
parser.add_argument("--n_total_episodes",type=int,default=200000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=20000,help="Save model every this number of episodes")
parser.add_argument("--record_data", type=bool, default=False, help="Whether to collect data while training.")
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/timing2d")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--len_delay", type=int, default=20, help="Number of timesteps in the delay period")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--len_edge", type=int, default=7, help="Length of the longer edge of the triangular arena. Must be odd number >=5")
parser.add_argument("--nonmatch_reward", type=int, default=100, help="Magnitude of reward when agent chooses nonmatch side")
parser.add_argument("--incorrect_reward", type=int, default=-100, help="Magnitude of reward when agent chooses match side")
parser.add_argument("--step_reward", type=float, default=-0.1, help="Magnitude of reward for each action agent takes, to ensure shortest path")
parser.add_argument("--poke_reward", type=int, default=5, help="Magnitude of reward when agent pokes signal to proceed")
parser.add_argument("--save_performance_fig", type=bool, default=False, help="If False, don't pass anything. If true, pass True.")
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
lr = argsdict['lr']
hidden_type = argsdict['hidden_type']
seed = argsdict['seed']
save_performance_fig = True if argsdict['save_performance_fig'] == True or argsdict['save_performance_fig'] == 'True' else False
ld = argsdict['len_delay']
len_edge = argsdict['len_edge']
rwd = argsdict['nonmatch_reward']
inc_rwd = argsdict['incorrect_reward']
step_rwd = argsdict['step_reward']
poke_rwd = argsdict['poke_reward']


# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/timing2d'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/timing2d'
save_dir = os.path.join(main_dir, f'{hidden_type}_{n_neurons}_{lr}')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(f'Saved to {save_dir}')

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])

# Define the environment
env = IntervalDiscrimination_2D(ld, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, seed=seed)

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
    batch_size=1,
    rfsize=rfsize,
    padding=padding,
    stride=stride)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
env_title = "Interval Discrimination 2D"
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

# Load existing model
if load_model_path=='None':
    ckpt_name = 'untrained_agent'  # placeholder ckptname in case we want to save data in the end
else:
    ckpt_name = load_model_path.replace('/', '_')
    # assert loaded model has congruent hidden type and n_neurons
    assert hidden_type in ckpt_name, 'Must load network with the same hidden type'
    assert str(n_neurons) in ckpt_name, 'Must load network with the same number of hidden neurons'
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/timing2d', load_model_path)))

# Initialize arrays for recording
if record_data:
    delay_resp_hx = np.zeros((n_total_episodes, 20, n_neurons), dtype=np.float32)  # hidden states during delay  # TODO: stim 1 and 2 should be upto 40, delay should be 20
    stim_1_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    stim_2_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
    delay_loc = np.zeros((n_total_episodes, 20, 2), dtype=np.int16)  # location during delay
    stim_1_loc = np.zeros((n_total_episodes, 40, 2), dtype=np.int16)
    stim_2_loc = np.zeros((n_total_episodes, 40, 2), dtype=np.int16)
    reward_hist = np.zeros((n_total_episodes, 20), dtype=np.int8)    # reward history

ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)
epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
correct_perc = np.zeros(n_total_episodes, dtype=np.float16)


for i_episode in tqdm(range(n_total_episodes)):
    done = False
    env.reset()
    ideal_nav_rwds[i_episode] = ideal_nav_rwd(env, step_rwd, poke_rwd, rwd)
    net.reinit_hid()
    if record_data:
        stim[i_episode] = [env.stim_1_len, env.stim_2_len]

    while not done:
        pol, val = net.forward(
            torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device)
        )  # forward
        if record_data and hidden_type=='lstm':
            if env.phase == "stim_1":
                stim_1_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_1_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            if env.phase == "stim_2":
                stim_2_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_2_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            if env.indelay:
                delay_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                delay_resp_hx[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
        elif record_data and hidden_type=='linear':
            if env.phase == "stim_1":
                stim_1_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_1_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()
            if env.phase == "stim_2":
                stim_2_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_2_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()
            if env.indelay:
                delay_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                delay_resp_hx[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()

        act, p, v = select_action(net, pol, val)
        new_obs, reward, done, info = env.step(act)
        net.rewards.append(reward)
        if env.indelay and record_data:
            reward_hist[i_episode, env.t-1] = reward

    if env.current_loc == env.correct_loc:  # env.current_loc == env.correct_loc
        correct_perc[i_episode] = 1
    epi_nav_reward[i_episode] = env.nav_reward

    p_loss, v_loss = finish_trial(net, 0.99, optimizer)

    if (i_episode+1) % save_ckpt_per_episodes == 0:
        print(f'Episode {i_episode}, {np.mean(correct_perc[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(correct_perc[:i_episode+1])*100:.3f}% correct')
        if save_ckpts:
            torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')

binned_correct_trial = bin_rewards(correct_perc, window_size)
fig, ax = plt.subplots()
fig.suptitle(env_title)
ax.plot(np.arange(n_total_episodes), binned_correct_trial, label=net_title)
ax.set_xlabel("Episode")
ax.set_ylabel("Correct rate")
ax.set_ylim(0,1)
ax.legend(frameon=False)
if save_performance_fig:
    fig.savefig(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance.svg')

# save data
if record_data:
    np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, delay_resp_hx=delay_resp_hx, delay_loc=delay_loc, reward_hist=reward_hist,
                        stim_1_resp=stim_1_resp, stim_1_loc=stim_1_loc, stim_2_resp=stim_2_resp, stim_2_loc=stim_2_loc,
                        epi_nav_reward=epi_nav_reward, ideal_nav_rwds=ideal_nav_rwds, correct_perc=correct_perc)
else:
    np.savez_compressed(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance_data.npz', epi_nav_reward=epi_nav_reward, ideal_nav_rwds=ideal_nav_rwds, correct_perc=correct_perc)
