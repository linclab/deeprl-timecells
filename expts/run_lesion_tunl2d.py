import random
import os
from re import I
from expts.agents.model_2d import *
from expts.envs.tunl_2d import Tunl
import numpy as np
import torch
import matplotlib.pyplot as plt
from lesion_expt_utils import generate_lesion_index
from analysis.linclab_utils import plot_utils
import argparse
from tqdm import tqdm

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



def lesion_experiment(env, net, optimizer, n_total_episodes, lesion_idx, save_dir, title, backprop=False):

    ct = np.zeros(n_total_episodes, dtype=np.int8)  # whether it's a correction trial or not
    stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
    epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
    nonmatch_perc = np.zeros(n_total_episodes, dtype=np.float16)
    choice = np.zeros((n_total_episodes, 2), dtype=np.int8)  # record the location when done
    ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)
    delay_loc = np.zeros((n_total_episodes, len_delay, 2), dtype=np.int16)  # location during delay
    delay_resp_hx = np.zeros((n_total_episodes, len_delay, n_neurons), dtype=np.float32)  # hidden states during delay
    delay_resp_cx = np.zeros((n_total_episodes, len_delay, n_neurons), dtype=np.float32)  # cell states during delay

    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        env.reset()
        ideal_nav_rwds[i_episode] = ideal_nav_rwd(env=env, len_edge=7, len_delay=env.len_delay, step_rwd=-0.1, poke_rwd=5)
        net.reinit_hid()
        stim[i_episode] = env.sample_loc
        ct[i_episode] = int(env.correction_trial)
        while not done:
            obs = torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device)
            pol, val = net.forward(obs, lesion_idx=lesion_idx)  # forward
            if env.indelay:  # record location and neural responses
                delay_loc[i_episode, env.delay_t - 1, :] = np.asarray(env.current_loc)
                delay_resp_hx[i_episode, env.delay_t - 1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
                delay_resp_cx[i_episode, env.delay_t - 1, :] = net.cx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            act, p, v = select_action(net, pol, val)
            new_obs, reward, done, info = env.step(act)
            net.rewards.append(reward)
        choice[i_episode] = env.current_loc
        if np.any(stim[i_episode] != choice[i_episode]):  # nonmatch
            nonmatch_perc[i_episode] = 1
        epi_nav_reward[i_episode] = env.nav_reward
        if backprop:
            p_loss, v_loss = finish_trial(net, 0.99, optimizer)

    torch.save(net.state_dict(), save_dir + f'/postlesion_{title}.pt')
    np.savez_compressed(save_dir + f'/lesion_{title}_data.npz', stim=stim, choice=choice, ct=ct, delay_loc=delay_loc,
                        delay_resp_hx=delay_resp_hx,
                        delay_resp_cx=delay_resp_cx,
                        epi_nav_reward=epi_nav_reward,
                        ideal_nav_rwds=ideal_nav_rwds)
    return np.mean(nonmatch_perc)


if __name__ == '__main__':

    plot_utils.linclab_plt_defaults()
    plot_utils.set_font(font='Helvetica')

    parser = argparse.ArgumentParser(description="lesion study in Non-location-fixed TUNL 2d task simulation")
    parser.add_argument("--n_total_episodes",type=int,default=1000,help="Total episodes to run lesion expt")
    parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl1d")
    parser.add_argument("--backprop", type=bool, default=False, help="Whether backprop loss during lesion expt")
    args = parser.parse_args()
    argsdict = args.__dict__
    print(argsdict)

    n_total_episodes = argsdict['n_total_episodes']
    load_model_path = argsdict['load_model_path']
    backprop = True if argsdict['backprop'] == True or argsdict['backprop'] == 'True' else False

    # Load_model_path: mem_40_lstm_256_5e-06/seed_1_epi999999.pt, relative to training/tunl2d
    config_dir = load_model_path.split('/')[0]
    data_name = load_model_path.replace('/', '_')+'_data.npz'
    data = np.load(os.path.join('/network/scratch/l/lindongy/timecell/data_collecting/tunl2d', config_dir, data_name), allow_pickle=True)

    # Load existing model
    ckpt_name = load_model_path.replace('/', '_').replace('.pt', '_pt')  # 'mem_40_lstm_256_5e-06_seed_1_epi999999_pt'
    hparams = ckpt_name.split('_')
    env_type = hparams[0]
    len_delay = int(hparams[1])
    hidden_type = hparams[2]
    n_neurons = int(hparams[3])
    lr = float(hparams[4])
    seed = int(hparams[6])
    epi = int(hparams[7][3:])
    assert hidden_type=='lstm', 'Lesion expt must be done on LSTM network'
    assert env_type=='mem', 'Lesion expt must be done on Mnemonic TUNL'
    env_title = 'Mnemonic TUNL 2D'
    net_title = 'LSTM'
    print(hidden_type, n_neurons, lr, seed, epi, env_title, net_title)

    stim = data["stim"]
    n_data_episodes = np.shape(stim)[0]
    print(f"Total number of episodes in saved data: {n_data_episodes}")
    if n_data_episodes > 5000:
        stim = stim[:-5000]
        choice = data["choice"][:-5000]
        delay_resp = data["delay_resp_hx"][:-5000]
        delay_loc = data["delay_loc"][:-5000]
    else:
        stim = stim
        choice = data["choice"]
        delay_resp = data["delay_resp_hx"]
        delay_loc = data["delay_loc"]

    # TODO: write a function to analyse cell_nums_ramp, cell_nums_seq, cell_nums_nontime
    # TODO: write a function to analyse stimulus-selective cell and readout cell

    # Make directory in /lesion to save data and model
    main_dir = '/network/scratch/l/lindongy/timecell/lesion/tunl2d'
    save_dir = os.path.join(main_dir, f'{env_type}_{len_delay}_{hidden_type}_{n_neurons}_{lr}_seed{seed}_epi{epi}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f'Saving to {save_dir}')

    # Setting up cuda and seeds
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = Tunl(len_delay, len_edge=7, rwd=100, inc_rwd=-20, step_rwd=-0.1, poke_rwd=5, rng_seed=seed)

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

    n_lesion = np.arange(start=0, stop=n_neurons, step=10)
    postlesion_perf_array = np.zeros((3, len(n_lesion)))

    for i_row, lesion_type in enumerate(['random', 'ramp', 'seq']):
        for i_column, num_lesion in enumerate(n_lesion):

            net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl2d', load_model_path)))
            net.eval()

            lesion_index = generate_lesion_index(lesion_type, num_lesion, n_neurons=n_neurons, cell_nums_ramp=cell_nums_ramp, cell_nums_seq=cell_nums_seq)

            postlesion_perf_array[i_row, i_column] = lesion_experiment(env=env, net=net, optimizer=optimizer,
                                                                       n_total_episodes=n_total_episodes,
                                                                       lesion_idx=lesion_index,
                                                                       title=f"{lesion_type}_{num_lesion}",
                                                                       save_dir=save_dir,
                                                                       backprop=backprop)
            print("Lesion type:", lesion_type, "; Lesion number:", num_lesion, "completed.")

    np.savez_compressed(os.path.join(save_dir, 'lesion_performance_arr.npz'), postlesion_perf=postlesion_perf_array)