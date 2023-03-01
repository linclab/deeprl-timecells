import gc
import random
import os
from re import I
from agents.model_1d import *
from envs.tunl_1d import TunlEnv
import numpy as np
import torch
import matplotlib.pyplot as plt
from lesion_expt_utils import generate_random_index
import argparse
from tqdm import tqdm
from numpy import array
import re
import sys
sys.path.insert(0, '..')
from analysis import utils_linclab_plot


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


def lesion_experiment(env, net, optimizer, n_total_episodes, lesion_idx, save_dir, title, save_net_and_data=False, backprop=False):
    stim = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
    nonmatch_perc = np.zeros(n_total_episodes, dtype=np.int8)
    first_action = np.zeros(n_total_episodes, dtype=np.int8)  # 0=L, 1=R
    delay_resp = np.zeros((n_total_episodes, len_delay, n_neurons), dtype=np.float32)

    for i_episode in tqdm(range(n_total_episodes)):  # one episode = one sample
        done = False
        env.reset()
        episode_sample = random.choices((array([[0, 0, 1, 0]]), array([[0, 0, 0, 1]])))[0]
        if np.all(episode_sample == array([[0, 0, 1, 0]])):  # L
            stim[i_episode] = 0
        elif np.all(episode_sample == array([[0, 0, 0, 1]])):  # R
            stim[i_episode] = 1
        resp = []
        net.reinit_hid()
        act_record = []
        while not done:
            pol, val, lin_act = net.forward(torch.as_tensor(env.observation).float().to(device), lesion_idx=lesion_idx)
            if np.all(env.observation == array([[0, 0, 0, 0]])) and env.delay_t>0:
                if net.hidden_types[1] == 'linear':
                    resp.append(net.cell_out[net.hidden_types.index("linear")].clone().detach().cpu().numpy().squeeze())  # pre-relu activity of first layer of linear cell
                elif net.hidden_types[1] == 'lstm':
                    resp.append(net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze())  # hidden state of LSTM cell
                elif net.hidden_types[1] == 'gru':
                    resp.append(net.hx[net.hidden_types.index("gru")].clone().detach().cpu().numpy().squeeze())  # hidden state of GRU cell
            act, p, v = select_action(net, pol, val)
            new_obs, reward, done = env.step(act, episode_sample)
            net.rewards.append(reward)
            act_record.append(act)
            del pol, val, lin_act, new_obs, p, v
        first_action[i_episode] = act_record[next((i for i, x in enumerate(net.rewards) if x), 0)] - 1  # The first choice that led to non-zero reward. 0=L, 1=R
        nonmatch_perc[i_episode] = 1 if stim[i_episode]+first_action[i_episode] == 1 else 0
        delay_resp[i_episode][:len(resp)] = np.asarray(resp)
        del resp, act_record
        if backprop:
            p_loss, v_loss = finish_trial(net, 0.99, optimizer)
        else:
            del net.rewards[:]
            del net.saved_actions[:]
    if save_net_and_data:
        net_and_data_dir = os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_net_and_data')
        os.mkdir(net_and_data_dir)
        torch.save(net.state_dict(), os.path.join(net_and_data_dir, f'/postlesion_{title}.pt'))
        np.savez_compressed(os.path.join(net_and_data_dir, f'/lesion_{title}_data.npz'), stim=stim, first_action=first_action, delay_resp=delay_resp)
    del stim, first_action, delay_resp
    return np.mean(nonmatch_perc)


if __name__ == '__main__':

    utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="../analysis/fonts")

    parser = argparse.ArgumentParser(description="lesion study in Head-fixed TUNL 1d task simulation")
    parser.add_argument("--n_total_episodes",type=int,default=1000,help="Total episodes to run lesion expt")
    parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/tunl1d_og")
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

    # Load_model_path: mem_40_lstm_256_0.0001/seed_1_epi999999.pt, relative to training/tunl1d_og
    config_dir = load_model_path.split('/')[0]
    pt_name = load_model_path.split('/')[1]
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    seed = int(pt[1])
    epi = int(pt[2])
    main_data_analysis_dir = '/network/scratch/l/lindongy/timecell/data_analysis/tunl1d_og'
    data_analysis_dir = os.path.join(main_data_analysis_dir, config_dir)
    ramp_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_ramp_ident_results_separate.npz'), allow_pickle=True)
    seq_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_seq_ident_results_separate.npz'), allow_pickle=True)
    cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
    cell_nums_seq = seq_ident_results['cell_nums_seq']

    # Load existing model
    ckpt_name = load_model_path.replace('/', '_').replace('.pt', '_pt')  # 'mem_40_lstm_256_0.0001_seed_1_epi999999_pt'
    hparams = config_dir.split('_')
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
    env_title = 'Mnemonic TUNL 1D'
    net_title = 'LSTM'
    print(hidden_type, n_neurons, lr, seed, epi, env_title, net_title)

    # Make directory in /lesion to save data and model
    main_dir = '/network/scratch/l/lindongy/timecell/lesion/tunl1d_og'
    save_dir = os.path.join(main_dir, f'{config_dir}_seed{seed}_epi{epi}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
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

    for i_lesion_type, lesion_type in enumerate(['random', 'ramp', 'seq']):
        for i_num_lesion, num_lesion in enumerate(n_lesion):
            for i_shuffle in tqdm(range(num_shuffle)):
                gc.collect()
                torch.cuda.empty_cache()
                env = TunlEnv(len_delay, seed=seed)
                if p is not None:
                    net = AC_Net(4, 4, 1, [hidden_type, 'linear'], [n_neurons, n_neurons], p_dropout=p, dropout_type=dropout_type)
                else:
                    net = AC_Net(4, 4, 1, [hidden_type, 'linear'], [n_neurons, n_neurons])
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/tunl1d_og', load_model_path)))
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
