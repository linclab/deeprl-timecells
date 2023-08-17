import random
import os
from re import I
from agents.model_1d import *
from envs.int_discrim import IntervalDiscrimination
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

parser = argparse.ArgumentParser(description="Head-fixed Interval Discrimination task simulation")
parser.add_argument("--n_total_episodes",type=int,default=50000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=5000,help="Save model every this number of episodes")
parser.add_argument("--record_data", type=bool, default=False, help="Whether to collect data while training.")
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/timing")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
# parser.add_argument("--len_delay", type=int, default=40, help="Number of timesteps in the delay period")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
# parser.add_argument("--env_type", type=str, default='mem', help="type of environment: mem or nomem")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--save_performance_fig", type=bool, default=False, help="If False, don't pass anything. If true, pass True.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--p_dropout", type=float, default=0.0, help="dropout probability")
parser.add_argument("--dropout_type", type=int, default=None, help="location of dropout (could be 1,2,3,or 4)")
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
# len_delay = argsdict['len_delay']
lr = argsdict['lr']
# env_type = argsdict['env_type']
hidden_type = argsdict['hidden_type']
seed = argsdict['seed']
save_performance_fig = True if argsdict['save_performance_fig'] == True or argsdict['save_performance_fig'] == 'True' else False
weight_decay = argsdict['weight_decay']
p_dropout = argsdict['p_dropout']
dropout_type = argsdict['dropout_type']

# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/timing'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/timing'
save_dir_str = f'{hidden_type}_{n_neurons}_{lr}'
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
torch.manual_seed(seed)
np.random.seed(seed)

env = IntervalDiscrimination(seed=seed)
net = AC_Net(
    input_dimensions=2,  # input dim
    action_dimensions=2,  # action dim
    hidden_types=[hidden_type, 'linear'],  # hidden types
    hidden_dimensions=[n_neurons, n_neurons], # hidden dims
    batch_size=1,
    p_dropout=p_dropout,
    dropout_type=dropout_type)
env_title = "Interval Discrimination"
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

# Load existing model
if load_model_path=='None':
    ckpt_name = f'seed_{seed}_untrained_agent'  # placeholder ckptname in case we want to save data in the end
else:
    ckpt_name = load_model_path.replace('/', '_')
    pt_name = load_model_path.split('/')[1]  # seed_3_epi199999.pt
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    loaded_ckpt_seed = int(pt[1])
    loaded_ckpt_episode = int(pt[2])
    # assert loaded model has congruent hidden type and n_neurons
    assert hidden_type in ckpt_name, 'Must load network with the same hidden type'
    assert str(n_neurons) in ckpt_name, 'Must load network with the same number of hidden neurons'
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/timing', load_model_path)))


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

# Train and record
# Initialize arrays for recording

action_hist = np.zeros(n_total_episodes, dtype=np.int8)
correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
if record_data:
    stim1_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    stim2_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    delay_resp = np.zeros((n_total_episodes, 20, n_neurons), dtype=np.float32)

# initialize hidden state dict for saving hidden states
hidden_state_dict = {}

for i_episode in tqdm(range(n_total_episodes)):
    done = False
    env.reset()
    if i_episode == 0:
        net.reinit_hid(saved_hidden=None)  # only initialize to 0 in the first episode
    else:
        net.reinit_hid(saved_hidden=hidden_state_dict)  # initialize to saved hidden state
    stim[i_episode,0] = env.first_stim
    stim[i_episode,1] = env.second_stim
    while not done:
        pol, val, lin_act = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device))  # forward
        if env.task_stage in ['init', 'choice_init']:
            act, p, v = select_action(net, pol, val)
            new_obs, reward, done = env.step(act)
            net.rewards.append(reward)
        else:
            new_obs, reward, done = env.step()

        if record_data and hidden_type=='lstm':
            if env.task_stage == 'first_stim' and env.elapsed_t > 0:
                stim1_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'second_stim' and env.elapsed_t > 0:
                stim2_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'delay' and env.elapsed_t > 0:
                delay_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
        elif record_data and hidden_type=='linear':
            if env.task_stage == 'first_stim' and env.elapsed_t > 0:
                stim1_resp[i_episode, env.elapsed_t-1, :] = net.cell_out[
                    net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'second_stim' and env.elapsed_t > 0:
                stim2_resp[i_episode, env.elapsed_t-1, :] = net.cell_out[
                    net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'delay' and env.elapsed_t > 0:
                delay_resp[i_episode, env.elapsed_t-1, :] = net.cell_out[
                    net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze()

        if env.task_stage == 'choice_init':
            action_hist[i_episode] = act
            correct_trial[i_episode] = env.correct_trial

    # Save hidden states for reinitialization in the next episode
    hidden_state_dict['cell_out'] = net.cell_out.clone().detach()
    hidden_state_dict['hx'] = net.hx.clone().detach()
    hidden_state_dict['cx'] = net.cx.clone().detach()

    if record_data: # No backprop
        del net.rewards[:]
        del net.saved_actions[:]
    else:
        p_loss, v_loss = finish_trial(net, 1, optimizer)
    if (i_episode+1) % save_ckpt_per_episodes == 0:
        if load_model_path != 'None':
            print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(correct_trial[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(correct_trial[:i_episode+1])*100:.3f}% correct')
        else:
            print(f'Episode {i_episode}, {np.mean(correct_trial[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(correct_trial[:i_episode+1])*100:.3f}% correct')
        if save_ckpts:
            if load_model_path != 'None':
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode+loaded_ckpt_episode}.pt')
            else:
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')

binned_correct_trial = bin_rewards(correct_trial, window_size)
fig, ax = plt.subplots()
fig.suptitle(env_title)
ax.plot(np.arange(n_total_episodes), binned_correct_trial, label=net_title)
ax.set_xlabel("Episode")
ax.set_ylabel("Correct rate")
ax.set_ylim(0,1)
ax.legend(frameon=False)
if save_performance_fig:
    fig.savefig(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance.svg')


if record_data:
    np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', action_hist=action_hist, correct_trial=correct_trial,
                        stim=stim, stim1_resp_hx=stim1_resp,
                        stim2_resp_hx=stim2_resp, delay_resp_hx=delay_resp)
else:
    np.savez_compressed(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance_data.npz', action_hist=action_hist, correct_trial=correct_trial, stim=stim)

