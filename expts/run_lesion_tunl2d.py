import random
import os
from agents.model_2d import *
from envs.tunl_2d import Tunl
import numpy as np
import torch
from lesion_expt_utils import generate_random_index
# from analysis.linclab_utils import plot_utils
import argparse
from tqdm import tqdm
import re
import gc
import matplotlib.pyplot as plt

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


def lesion_experiment(env, net, optimizer, n_total_episodes, lesion_idx, save_dir, title, save_net_and_data=False, backprop=False):

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
    if save_net_and_data:
        net_and_data_dir = os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_net_and_data')
        os.mkdir(net_and_data_dir)
        torch.save(net.state_dict(), os.path.join(net_and_data_dir, f'/postlesion_{title}.pt'))
        np.savez_compressed(os.path.join(net_and_data_dir, f'/lesion_{title}_data.npz'), stim=stim, choice=choice, ct=ct, delay_loc=delay_loc,
                            delay_resp_hx=delay_resp_hx,
                            delay_resp_cx=delay_resp_cx,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds)
    return np.mean(nonmatch_perc)


if __name__ == '__main__':

    # plot_utils.linclab_plt_defaults()
    # plot_utils.set_font(font='Helvetica')

    parser = argparse.ArgumentParser(description="lesion study in Non-location-fixed TUNL 2d task simulation")
    parser.add_argument("--n_total_episodes",type=int,default=1000,help="Total episodes to run lesion expt")
    parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl2d")
    parser.add_argument("--backprop", type=bool, default=False, help="Whether backprop loss during lesion expt")
    parser.add_argument("--save_net_and_data", type=bool, default=False, help="Save hx during lesion expt and save net after lesion expt")
    parser.add_argument("--num_shuffle", type=int, default=100, help="Number of times to shuffle the neuron indices, one shuffle = one lesion expt")
    parser.add_argument("--verbose", type=bool, default=False, help="if True, print nonmatch performance of each lesion expt")
    parser.add_argument("--lesion_idx_start", type=int, default=5, help="start of lesion index")
    parser.add_argument("--lesion_idx_end", type=int, default=None, help="end of lesion index. if None (default), end at n_neurons")
    parser.add_argument("--lesion_idx_step", type=int, default=5, help="step of lesion index")
    args = parser.parse_args()
    argsdict = args.__dict__
    print(argsdict)

    n_total_episodes = argsdict['n_total_episodes']
    load_model_path = argsdict['load_model_path']
    backprop = True if argsdict['backprop'] == True or argsdict['backprop'] == 'True' else False
    verbose = True if argsdict['verbose'] == True or argsdict['verbose'] == 'True' else False
    save_net_and_data = True if argsdict['save_net_and_data'] == True or argsdict['save_net_and_data'] == 'True' else False
    num_shuffle = argsdict['num_shuffle']
    lesion_idx_start = argsdict['lesion_idx_start']
    lesion_idx_step = argsdict['lesion_idx_step']
    lesion_idx_end = argsdict['lesion_idx_end']

    # Load_model_path: mem_40_lstm_256_5e-06/seed_1_epi999999.pt, relative to training/tunl2d
    config_dir = load_model_path.split('/')[0]
    pt_name = load_model_path.split('/')[1]
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    seed = int(pt[1])
    epi = int(pt[2])
    main_data_analysis_dir = '/network/scratch/l/lindongy/timecell/data_analysis/tunl2d_100_99.9'
    data_analysis_dir = os.path.join(main_data_analysis_dir, config_dir)
    ramp_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_ramp_ident_results_separate.npz'), allow_pickle=True)
    seq_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_seq_ident_results_separate.npz'), allow_pickle=True)
    cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
    cell_nums_seq = seq_ident_results['cell_nums_seq']

    # Load existing model
    ckpt_name = load_model_path.replace('/', '_').replace('.pt', '_pt')  # 'mem_40_lstm_256_5e-06_seed_1_epi999999_pt'
    hparams = ckpt_name.split('_')
    env_type = hparams[0]
    len_delay = int(hparams[1])
    hidden_type = hparams[2]
    n_neurons = int(hparams[3])
    lr = float(hparams[4])
    wd = None
    p = None
    dropout_type = None
    if len(hparams) > 5:  # weight_decay or dropout
        if 'wd' in hparams[5]:
            wd = float(hparams[5][2:])
        if 'p' in hparams[5]:
            p = float(hparams[5][1:])
            dropout_type = hparams[6]
    assert hidden_type=='lstm', 'Lesion expt must be done on LSTM network'
    assert env_type=='mem', 'Lesion expt must be done on Mnemonic TUNL'
    env_title = 'Mnemonic TUNL 2D'
    net_title = 'LSTM'
    print(hidden_type, n_neurons, lr, seed, epi, env_title, net_title)


    # Make directory in /lesion to save data and model
    main_dir = '/network/scratch/l/lindongy/timecell/lesion/tunl2d'
    save_dir = os.path.join(main_dir, f'{config_dir}_seed{seed}_epi{epi}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Saving to {save_dir}')

    # Setting up cuda and seeds
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if lesion_idx_end is not None:  # specified lesion_idx_end
        n_lesion = np.arange(start=lesion_idx_start, stop=lesion_idx_end, step=lesion_idx_step)
    else:
        n_lesion = np.arange(start=lesion_idx_start, stop=n_neurons, step=lesion_idx_step)

    postlesion_perf_array = np.zeros((3, len(n_lesion), num_shuffle))

    random_index_dict = generate_random_index(num_shuffle, n_neurons, cell_nums_ramp, cell_nums_seq)

    rfsize = 2
    padding = 0
    stride = 1
    dilation = 1
    conv_1_features = 16
    conv_2_features = 32

    # Define conv & pool layer sizes
    layer_1_out_h, layer_1_out_w = conv_output(6, 9, padding, dilation, rfsize, stride)  # env.h=6, env.w=9
    layer_2_out_h, layer_2_out_w = conv_output(layer_1_out_h, layer_1_out_w, padding, dilation, rfsize, stride)
    layer_3_out_h, layer_3_out_w = conv_output(layer_2_out_h, layer_2_out_w, padding, dilation, rfsize, stride)
    layer_4_out_h, layer_4_out_w = conv_output(layer_3_out_h, layer_3_out_w, padding, dilation, rfsize, stride)

    for i_lesion_type, lesion_type in enumerate(['random', 'ramp', 'seq']):
        for i_num_lesion, num_lesion in enumerate(n_lesion):
            for i_shuffle in tqdm(range(num_shuffle)):
                gc.collect()
                torch.cuda.empty_cache()
                env = Tunl(len_delay, len_edge=7, rwd=100, inc_rwd=-20, step_rwd=-0.1, poke_rwd=5, rng_seed=seed)
                if p is None:
                    net = AC_Net(
                        input_dimensions=(6, 9, 3),  # input dim
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
                else:
                    net = AC_Net(
                        input_dimensions=(6, 9, 3),  # input dim
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
                        stride=stride,
                        p_dropout=p,
                        dropout_type=dropout_type)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl2d', load_model_path)))
                net.eval()

                lesion_index = random_index_dict[lesion_type][i_shuffle][:num_lesion]

                postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle] = lesion_experiment(env=env, net=net, optimizer=optimizer,
                                                                                                  n_total_episodes=n_total_episodes,
                                                                                                  lesion_idx=lesion_index,
                                                                                                  title=f"{lesion_type}_{num_lesion}",
                                                                                                  save_dir=save_dir, save_net_and_data=save_net_and_data,
                                                                                                  backprop=backprop)
                if verbose:
                    print(f"Lesion type: {lesion_type} ; Lesion number: {num_lesion} ; completed. {postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle]*100:.3f}% nonmatch")
                del env, net, optimizer

    fig, ax1 = plt.subplots()
    fig.suptitle(f'{env_title}_{ckpt_name}')
    ax1.plot(n_lesion, np.mean(postlesion_perf_array[0, :, :], axis=-1), color='gray', label='Random lesion')
    ax1.fill_between(n_lesion,
                     np.mean(postlesion_perf_array[0, :, :], axis=-1)-np.std(postlesion_perf_array[0, :, :], axis=-1),
                     np.mean(postlesion_perf_array[0, :, :], axis=-1)+np.std(postlesion_perf_array[0, :, :], axis=-1), color='lightgray', alpha=0.2)
    ax1.plot(n_lesion, np.mean(postlesion_perf_array[1, :, :], axis=-1), color='royalblue', label='Ramping cell lesion')
    ax1.fill_between(n_lesion,
                     np.mean(postlesion_perf_array[1, :, :], axis=-1)-np.std(postlesion_perf_array[1, :, :], axis=-1),
                     np.mean(postlesion_perf_array[1, :, :], axis=-1)+np.std(postlesion_perf_array[1, :, :], axis=-1), color='lightsteelblue', alpha=0.2)
    ax1.plot(n_lesion, np.mean(postlesion_perf_array[2, :, :], axis=-1), color='magenta', label='Time cell lesion')
    ax1.fill_between(n_lesion,
                     np.mean(postlesion_perf_array[2, :, :], axis=-1)-np.std(postlesion_perf_array[2, :, :], axis=-1),
                     np.mean(postlesion_perf_array[2, :, :], axis=-1)+np.std(postlesion_perf_array[2, :, :], axis=-1), color='thistle', alpha=0.2)
    ax1.hlines(y=0.5, linestyles='dashed', colors='gray')
    ax1.set_xlabel('Number of neurons lesioned')
    ax1.set_ylabel('Fraction Nonmatch')
    ax1.set_ylim(0,1)
    ax1.legend()
    #plt.show()
    fig.savefig(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_lesion_results.png'))

    np.savez_compressed(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_lesion_results.npz'), random_index_dict=random_index_dict, n_lesion=n_lesion, postlesion_perf=postlesion_perf_array)

    np.savez_compressed(os.path.join(save_dir, 'lesion_performance_arr.npz'), postlesion_perf=postlesion_perf_array)
