import random
import torch
import numpy as np
from numpy import array
from envs.tunl_1d import TunlEnv_dim2, TunlEnv_nomem_dim2
from agents.model_1d import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import re
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

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
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl1d_og")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--len_delay", type=int, default=40, help="Number of timesteps in the delay period")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
parser.add_argument("--env_type", type=str, default='mem', help="type of environment: mem or nomem")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--save_performance_fig", type=bool, default=False, help="If False, don't pass anything. If true, pass True.")
parser.add_argument("--p_dropout", type=float, default=0.0, help="dropout probability")
parser.add_argument("--dropout_type", type=int, default=None, help="location of dropout (could be 1,2,3,or 4)")
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
#env_type = argsdict['env_type']
env_type = 'nomem'
hidden_type = argsdict['hidden_type']
seed = argsdict["seed"]
weight_decay = argsdict['weight_decay']
p_dropout = argsdict['p_dropout']
dropout_type = argsdict['dropout_type']
# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl1d_og_dim2/timing_pretrained'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/tunl1d_og_dim2/timing_pretrained'
save_dir_str = f'{env_type}_{len_delay}_{hidden_type}_{n_neurons}_{lr}'
if weight_decay != 0:
    save_dir_str += f'_wd{weight_decay}'
if p_dropout != 0:
    save_dir_str += f'_p{p_dropout}_{dropout_type}'
save_dir = os.path.join(main_dir, save_dir_str)
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir, exist_ok=True)
#print(f'Saved to {save_dir}')

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = TunlEnv_dim2(len_delay, seed=seed) if env_type=='mem' else TunlEnv_nomem_dim2(len_delay, seed=seed)
net = AC_Net(2, 2, 1, [hidden_type, 'linear'], [n_neurons, n_neurons], p_dropout, dropout_type)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
env_title = 'Mnemonic' if env_type == 'mem' else 'Non-mnemonic'
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'

# Load existing model
if load_model_path=='None':
    ckpt_name = f'seed_{seed}_untrained_agent_weight_frozen'  # placeholder ckptname in case we want to save data in the end
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f'Saved to {save_dir}')
else:
    ckpt_name = load_model_path.replace('/', '_')
    pt_name = load_model_path.split('/')[1]  # seed_3_epi199999.pt
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    loaded_ckpt_seed = int(pt[1])
    loaded_ckpt_episode = int(pt[2])
    # assert loaded model has congruent hidden type and n_neurons
    assert hidden_type in ckpt_name, 'Must load network with the same hidden type'
    assert str(n_neurons) in ckpt_name, 'Must load network with the same number of hidden neurons'
    #net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl1d_og_dim2', load_model_path)))
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/timing', load_model_path)))
    save_dir = os.path.join(save_dir, f'model_{loaded_ckpt_seed}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f'Saved to {save_dir}')

stim = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
nonmatch_perc = np.zeros(n_total_episodes, dtype=np.int8)
first_action = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
nomem_perf = np.zeros(n_total_episodes, dtype=np.int8)
if record_data:
    delay_resp = np.zeros((n_total_episodes, len_delay, n_neurons), dtype=np.float32)

for i_episode in tqdm(range(n_total_episodes)):  # one episode = one sample
    #print(f"=========================== EPISODE {i_episode} ==============================")
    done = False
    episode_sample = random.choices((array([[1, 0]]), array([[0, 1]])))[0]
    env.reset(episode_sample)  # observation is set to episode_sample
    if np.all(episode_sample == array([[1, 0]])):  # L
        stim[i_episode] = 0
    elif np.all(episode_sample == array([[0, 1]])):  # R
        stim[i_episode] = 1
    if record_data:
        resp = []
    net.reinit_hid()
    act_record = []
    while not done:
        #print('obs: ', env.observation)
        #print('sample: ', env.sample)
        pol, val, lin_act = net.forward(torch.as_tensor(env.observation).float().to(device), temperature=1.5)
        if np.all(env.observation == array([[0, 0]])) and env.delay_t>0:
            if record_data:
                if hidden_type == 'linear':
                    resp.append(net.cell_out[net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze())  # pre-relu activity of first layer of linear cell
                elif hidden_type == 'lstm':
                    resp.append(net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())  # hidden state of LSTM cell
                elif hidden_type == 'gru':
                    resp.append(net.hx[net.hidden_types.index("gru")].clone().detach().cpu().numpy().squeeze())  # hidden state of GRU cell
                elif hidden_type == 'rnn':
                    resp.append(net.hx[net.hidden_types.index("rnn")].clone().detach().cpu().numpy().squeeze())
        act, p, v = select_action(net, pol, val)
        new_obs, reward, done = env.step(act)
        #print('act: ', act)
        #print('reward: ', reward)
        net.rewards.append(reward)
        act_record.append(act)
    first_action[i_episode] = act_record[next((i for i, x in enumerate(net.rewards) if x), 0)]  # The first choice that led to non-zero reward. 0=L, 1=R
    nonmatch_perc[i_episode] = 1 if stim[i_episode]+first_action[i_episode] == 1 else 0
    nomem_perf[i_episode] = reward
    if record_data:
        delay_resp[i_episode][:len(resp)] = np.asarray(resp)
        # for untrained agent, freeze weight
        del net.rewards[:]
        del net.saved_actions[:]
    else:
        p_loss, v_loss = finish_trial(net, 0.99, optimizer)
    if (i_episode+1) % save_ckpt_per_episodes == 0:
        if load_model_path != 'None':
            #print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(nonmatch_perc[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% nonmatch in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nonmatch_perc[:i_episode+1])*100:.3f}% nonmatch')
            #print(f'Episode {i_episode+loaded_ckpt_episode}, {np.mean(nomem_perf[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nomem_perf[:i_episode+1])*100:.3f}% correct')
            print(f'Episode {i_episode}, {np.mean(nomem_perf[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nomem_perf[:i_episode+1])*100:.3f}% correct')
        else:
            #print(f'Episode {i_episode}, {np.mean(nonmatch_perc[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% nonmatch in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nonmatch_perc[:i_episode+1])*100:.3f}% nonmatch')
            print(f'Episode {i_episode}, {np.mean(nomem_perf[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(nomem_perf[:i_episode+1])*100:.3f}% correct')
        if save_ckpts:
            if load_model_path != 'None':
                #torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode+loaded_ckpt_episode}.pt')
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')
            else:
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')
#binned_nonmatch_perc = bin_rewards(nonmatch_perc, window_size=window_size)
binned_reward = bin_rewards(nomem_perf, window_size=window_size)

fig, ax1 = plt.subplots()
fig.suptitle(f'{env_title} TUNL')
#ax1.plot(np.arange(n_total_episodes), binned_nonmatch_perc, label=net_title)
ax1.plot(np.arange(n_total_episodes), binned_reward, label=net_title)
ax1.set_xlabel('Episode')
ax1.set_ylabel('% Correct')
ax1.set_ylim(0,1)
ax1.legend()
#plt.show()
if save_performance_fig:
    fig.savefig(save_dir + f'/seed_{argsdict["seed"]}_total_{n_total_episodes}episodes_performance.svg')

# save data
if record_data:
    #np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, first_action=first_action, delay_resp=delay_resp)
    np.savez_compressed(save_dir + f'/{ckpt_name}_data.npz', stim=stim, first_action=first_action, nomem_perf=nomem_perf, delay_resp=delay_resp)
else:
    #np.savez_compressed(save_dir + f'/seed_{argsdict["seed"]}_total_{n_total_episodes}episodes_performance_data.npz', stim=stim, first_action=first_action)
    np.savez_compressed(save_dir + f'/seed_{argsdict["seed"]}_total_{n_total_episodes}episodes_performance_data.npz', stim=stim, first_action=first_action, nomem_perf=nomem_perf)

#del stim, nonmatch_perc, first_action, net, env, optimizer
del stim, nonmatch_perc, nomem_perf, first_action, net, env, optimizer
if record_data:
    del delay_resp


