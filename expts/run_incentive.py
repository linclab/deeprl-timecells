import random
import os
from envs.tunl_2d import run_to_reward_port
from agents.model_2d import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import re
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")
from PIL import Image
import gc

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


def ideal_nav_rwd(env, len_edge, step_rwd, poke_rwd):
    """
    Given env, len_edge, len_delay, step_rwd, poke_rwd, return the ideal navigation reward for a single episode.
    Use after env.reset().
    """
    ideal_nav_reward = (env.dist_to_sample + (len_edge - 1)) * step_rwd + poke_rwd
    return ideal_nav_reward


parser = argparse.ArgumentParser(description="Non-location-fixed TUNL 2D task simulation")
parser.add_argument("--n_total_episodes",type=int,default=50000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=5000,help="Save model every this number of episodes")
parser.add_argument("--record_data", type=bool, default=False, help="Whether to collect data while training. If False, don't pass anything. If true, pass True.")
parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/td_incentive")
parser.add_argument("--save_ckpts", type=bool, default=False, help="Whether to save model every save_ckpt_per_epidoes episodes. If False, don't pass anything. If true, pass True.")
parser.add_argument("--n_neurons", type=int, default=512, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--seed", type=int, default=1, help="seed to ensure reproducibility")
parser.add_argument("--hidden_type", type=str, default='lstm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--len_edge", type=int, default=7, help="Length of the longer edge of the triangular arena. Must be odd number >=5")
parser.add_argument("--step_reward", type=float, default=-0.1, help="Magnitude of reward for each action agent takes, to ensure shortest path")
parser.add_argument("--poke_reward", type=float, default=1, help="Magnitude of reward when agent pokes signal to proceed")
parser.add_argument("--incentive_mag", type=float, default=10, help="Magnitude of reward when agent moves to reward port")
parser.add_argument("--incentive_prob", type=float, default=1, help="Number of steps before agent can receive incentive reward")
parser.add_argument("--save_performance_fig", type=bool, default=False, help="If False, don't pass anything. If true, pass True.")
parser.add_argument("--algo", type=str, default='td', help="td or mc")
parser.add_argument("--truncate_step", type=int, default=200, help="truncate BPTT frequency")
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
lr = argsdict['lr']
hidden_type = argsdict['hidden_type']
len_edge = argsdict['len_edge']
step_reward = argsdict['step_reward']
poke_reward = argsdict['poke_reward']
seed = argsdict['seed']
incentive_mag = argsdict['incentive_mag']
incentive_prob = argsdict['incentive_prob']
algo = argsdict['algo']
truncate_step = argsdict['truncate_step']

if record_data:
    main_dir = '/network/scratch/l/lindongy/timecell/data_collecting/td_incentive'
else:
    main_dir = '/network/scratch/l/lindongy/timecell/training/td_incentive'
save_dir_str = f'{hidden_type}_N{n_neurons}_{lr}_{algo}_truncate{truncate_step}_len{len_edge}_R{incentive_mag}_P{incentive_prob}'
save_dir = os.path.join(main_dir, save_dir_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
print(f'Saved to {save_dir}')

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])
random.seed(argsdict["seed"])
if use_cuda:
    torch.cuda.manual_seed_all(argsdict["seed"])

env = run_to_reward_port(len_edge=len_edge,
                         step_reward=step_reward,
                         poke_reward=poke_reward,
                         incentive_magnitude=incentive_mag,
                         incentive_probability=incentive_prob,
                         rng_seed=seed)

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
net_title = 'LSTM' if hidden_type == 'lstm' else 'Feedforward'
net_title += f"_{algo}"

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
    net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/td_incentive', load_model_path)))

stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
epi_incentive_reward = np.zeros(n_total_episodes, dtype=np.float16)
ideal_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
p_losses = np.zeros(n_total_episodes, dtype=np.float16)
v_losses = np.zeros(n_total_episodes, dtype=np.float16)
n_steps = np.zeros(n_total_episodes, dtype=np.int8)
if record_data:  # list of lists. Each sublist is data from one episode
    neural_activity = []
    action = []
    location = []
    rwd = []

render = False

# initialize hidden state dict for saving hidden states
hidden_state_dict = {}

# Training loop
for i_episode in tqdm(range(n_total_episodes)):
    torch.cuda.empty_cache()
    gc.collect()
    done = False
    env.reset()
    stim[i_episode, :] = env.sample_loc
    ideal_nav_reward[i_episode] = ideal_nav_rwd(env, len_edge, step_reward, poke_reward)
    net.reinit_hid(saved_hidden=None)  # initialize to 0 at the beginning of each episode
    step = 0
    if record_data:
        neural_activity.append([])
        action.append([])
        location.append([])
        rwd.append([])

    while not done:

        # Render the observation as an image in the notebook
        if render:
            # Convert observation to PIL image
            image = Image.fromarray(env.observation.astype('uint8'), 'RGB')
            plt.imshow(image)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.title(f"Episode {i_episode + 1}, Step {step + 1}")
            plt.pause(0.01)  # Briefly pause to update the display

        obs = torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device)
        pol, val = net.forward(obs)  # forward pass
        act, p, v = select_action(net, pol, val)
        if record_data:
            location[i_episode].append(env.current_loc)
            neural_activity[i_episode].append(net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())
            action[i_episode].append(act)
        new_obs, reward, done, info = env.step(act)
        if record_data:
            rwd[i_episode].append(reward)
        net.rewards.append(reward)
        step += 1

        if algo == 'td' and (step+1) % truncate_step == 0:
            # Save hidden states for reinitialization in the next episode
            with torch.no_grad():
                hidden_state_dict['cell_out'] = [x.clone().detach() if x!=None else None for x in net.cell_out]
                hidden_state_dict['hx'] = [x.clone().detach() if x!=None else None for x in net.hx]
                hidden_state_dict['cx'] = [x.clone().detach() if x!=None else None for x in net.cx]
            p_loss, v_loss = finish_trial_td(net, 0.99, optimizer)
            net.reinit_hid(hidden_state_dict)

    epi_nav_reward[i_episode] = env.nav_reward
    epi_incentive_reward[i_episode] = env.reward
    n_steps[i_episode] = step
    print(f"Episode {i_episode}, {step} steps")
    if record_data:
        del net.rewards[:]
        del net.saved_actions[:]
    else:
        if algo == 'td':
            if (step+1) % truncate_step != 0:
                p_losses[i_episode], v_losses[i_episode] = finish_trial_td(net, 0.99, optimizer)
        elif algo == 'mc':
            p_losses[i_episode], v_losses[i_episode] = finish_trial_mc(net, 0.99, optimizer)
        else:
            raise ValueError('algo must be td or mc')
    if (i_episode+1) % save_ckpt_per_episodes == 0:
        if load_model_path != 'None':
            print(f'Episode {i_episode+1+loaded_ckpt_episode}')
        else:
            print(f'Episode {i_episode+1}')
        print(f'Average navigation reward {np.mean(epi_nav_reward[i_episode+1-save_ckpt_per_episodes:i_episode+1])} out of {np.mean(ideal_nav_reward[i_episode+1-save_ckpt_per_episodes:i_episode+1])} in the last {save_ckpt_per_episodes} episodes')
        print(f'Total average navigation reward {np.mean(epi_nav_reward[:i_episode+1])} out of {np.mean(ideal_nav_reward[:i_episode+1])}')
        if save_ckpts:
            if load_model_path != 'None':
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode+loaded_ckpt_episode}.pt')
            else:
                torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')


# Save data
if load_model_path != 'None':
    save_performance_name = f'seed_{argsdict["seed"]}_epi{loaded_ckpt_episode}_to_{n_total_episodes+loaded_ckpt_episode}_performance_data.npz'
else:
    save_performance_name = f'seed_{argsdict["seed"]}_epi0_to_{n_total_episodes}episodes_performance_data.npz'

np.savez_compressed(
    os.path.join(save_dir, save_performance_name),
                 stim=stim,
                 epi_nav_reward=epi_nav_reward,
                 epi_incentive_reward=epi_incentive_reward,
                 ideal_nav_reward=ideal_nav_reward,
                 p_losses=p_losses,
                v_losses=v_losses,
                n_steps=n_steps)

if record_data:
    if load_model_path != 'None':
        save_data_name = f'seed_{argsdict["seed"]}_epi{loaded_ckpt_episode}_to_{n_total_episodes+loaded_ckpt_episode}_neural_data.npz'
    else:
        save_data_name = f'seed_{argsdict["seed"]}_epi0_to_{n_total_episodes}episodes_neural_data.npz'

    np.savez_compressed(
        os.path.join(save_dir, save_data_name),
        neural_activity=neural_activity,
        action=action,
        location=location,
        reward=rwd)
