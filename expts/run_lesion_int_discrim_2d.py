import os
import random
from agents.model_2d import *
from envs.int_discrim_2d import IntervalDiscrimination_2D
import numpy as np
import torch
from lesion_expt_utils import generate_random_index
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
import argparse
import re
from tqdm import tqdm
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


def ideal_nav_rwd(env, step_rwd, poke_rwd, rwd):
    """
    Given env, len_edge, len_delay, step_rwd, poke_rwd, return the ideal navigation reward for a single episode.
    Use after env.reset().
    """
    ideal_nav_reward = env.dist_to_init * step_rwd + poke_rwd + rwd
    return ideal_nav_reward


def lesion_experiment(env, net, optimizer, n_total_episodes, lesion_idx, title, save_dir, backprop=False, save_net_and_data=False):
    delay_resp = np.zeros((n_total_episodes, 20, n_neurons), dtype=np.float32)  # hidden states during delay  # TODO: stim 1 and 2 should be upto 40, delay should be 20
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
        net.reinit_hid()
        ideal_nav_rwds[i_episode] = ideal_nav_rwd(env, env.step_rwd, env.poke_rwd, env.rwd)
        net.reinit_hid()
        stim[i_episode] = [env.stim_1_len, env.stim_2_len]
        while not done:
            pol, val = net.forward(
                torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device),
                lesion_idx=lesion_idx
            )
            if env.phase == "stim_1":
                stim_1_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_1_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            if env.phase == "stim_2":
                stim_2_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                stim_2_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            if env.indelay:
                delay_loc[i_episode, env.t - 1, :] = np.asarray(env.current_loc)
                delay_resp[i_episode, env.t - 1, :] = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            act, p, v = select_action(net, pol, val)
            new_obs, reward, done, info = env.step(act)
            net.rewards.append(reward)
        if env.current_loc == env.correct_loc:  # env.current_loc == env.correct_loc
            correct_perc[i_episode] = 1
        epi_nav_reward[i_episode] = env.nav_reward

        if backprop:
            p_loss, v_loss = finish_trial(net, 1, optimizer)
        else:
            del net.rewards[:]
            del net.saved_actions[:]

    if save_net_and_data:
        net_and_data_dir = os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_net_and_data')
        os.mkdir(net_and_data_dir)
        #torch.save(net.state_dict(), os.path.join(net_and_data_dir, f'postlesion_{title}.pt'))
        np.savez_compressed(os.path.join(net_and_data_dir, f'lesion_{title}_data.npz'), reward_hist=reward_hist, correct_perc=correct_perc,
                            stim=stim, stim1_resp_hx=stim_1_resp,
                            stim2_resp_hx=stim_2_resp, delay_resp_hx=delay_resp)
    del reward_hist, stim, stim_1_resp, stim_2_resp, delay_resp
    return np.mean(correct_perc), np.mean(epi_nav_reward)


def rehydration_experiment(env, net, n_total_episodes, lesion_idx):
    # new policy is calculated from silencing lesion_idx WHILE other hx stay the same
    reward_hist = np.zeros((n_total_episodes, 20), dtype=np.int8)    # reward history
    reward_hist_prime = np.zeros((n_total_episodes, 20), dtype=np.int8)
    ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)
    epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
    epi_nav_reward_prime = np.zeros(n_total_episodes, dtype=np.float16)
    correct_perc = np.zeros(n_total_episodes, dtype=np.float16)
    correct_perc_prime = np.zeros(n_total_episodes, dtype=np.float16)
    stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
    kl_div = []
    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        env.reset()
        net.reinit_hid()
        ideal_nav_rwds[i_episode] = ideal_nav_rwd(env, env.step_rwd, env.poke_rwd, env.rwd)
        net.reinit_hid()
        stim[i_episode] = [env.stim_1_len, env.stim_2_len]
        nav_reward_prime = 0
        pi_record = []
        pi_prime_record = []
        action_record = []
        action_prime_record = []
        while not done:
            pol, val = net.forward(
                torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device),
            )
            pi_record.append(pol)
            new_activity = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            new_activity[lesion_idx] = 0  # the new, manipulated hx
            lin_out = F.relu(net.hidden[net.hidden_types.index("linear")](torch.from_numpy(new_activity).to(device)))
            new_pol = F.softmax(net.output[0](lin_out), dim=0)
            pi_prime_record.append(new_pol)

            action = Categorical(pol).sample().item()
            action_record.append(action)
            new_action = Categorical(new_pol).sample().item()
            action_prime_record.append(new_action)

            # proceed with trial with old policy
            new_obs, reward, done, _ = env.step(action)
            # calculate hypothetical reward if the new action from rehydrated network was taken
            reward_prime = env.calc_reward_without_stepping(new_action)
            if reward_prime == env.step_rwd or reward_prime == env.poke_rwd or reward_prime == 0:  # navigation reward
                nav_reward_prime += reward_prime
            if env.current_loc == env.correct_loc:  # env.current_loc == env.correct_loc
                correct_perc[i_episode] = 1
            if reward_prime == env.rwd:
                correct_perc_prime[i_episode] = 1
        epi_nav_reward[i_episode] = env.nav_reward
        epi_nav_reward_prime[i_episode] = nav_reward_prime
        # calculate KL divergence between original policy pi and new policy pi'
        kl_divergence = 0
        for pi, pi_p in zip(pi_record, pi_prime_record):
            kl_divergence += F.kl_div(pi_p, pi, reduction='batchmean')  # TODO: try removing .log()
        kl_divergence /= len(pi_record)
        kl_div.append(kl_divergence.item())
        # print('KL divergence:', kl_divergence.item())
    del stim, reward_hist
    return np.mean(correct_perc), np.mean(correct_perc_prime), np.mean(epi_nav_reward), np.mean(epi_nav_reward_prime), np.mean(np.asarray(kl_div))


if __name__ == '__main__':

    utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

    parser = argparse.ArgumentParser(description="lesion study in non-location-fixed Interval Discrimination task simulation")
    parser.add_argument("--n_total_episodes",type=int,default=1000,help="Total episodes to run lesion expt")
    parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/timing")
    parser.add_argument("--backprop", type=bool, default=False, help="Whether backprop loss during lesion expt")
    parser.add_argument("--save_net_and_data", type=bool, default=False, help="Save hx during lesion expt and save net after lesion expt")
    parser.add_argument("--num_shuffle", type=int, default=100, help="Number of times to shuffle the neuron indices, one shuffle = one lesion expt")
    parser.add_argument("--verbose", type=bool, default=False, help="if True, print nonmatch performance of each lesion expt")
    parser.add_argument("--lesion_idx_start", type=int, default=5, help="start of lesion index")
    parser.add_argument("--lesion_idx_end", type=int, default=None, help="end of lesion index. if None (default), end at n_neurons")
    parser.add_argument("--lesion_idx_step", type=int, default=5, help="step of lesion index")
    parser.add_argument("--expt_type", type=str, default='lesion', help='lesion or rehydration')
    parser.add_argument("--n_ramp_time_shuffle", type=int, default=100, help="Number of shuffles used for identifying ramping and time cells")
    parser.add_argument("--ramp_time_percentile", type=float, default=99.9, help="Percentile at which ramping and time cells are identified")
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
    n_ramp_time_shuffle = argsdict['n_ramp_time_shuffle']
    ramp_time_percentile = argsdict['ramp_time_percentile']

    # Load_model_path: lstm_256_5e-06/seed_1_epi149999.pt, relative to training/timing
    config_dir = load_model_path.split('/')[0]
    pt_name = load_model_path.split('/')[1]
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    seed = int(pt[1])
    epi = int(pt[2])
    main_dir = '/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/analysis_results/timing2d'
    data_analysis_dir = os.path.join(main_dir, config_dir)

    cell_nums_ramp = []
    cell_nums_time = []
    for label in ['stimulus_1', 'stimulus_2']:
        print(f"analysing data from {label}")
        ramp_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_{n_ramp_time_shuffle}_{ramp_time_percentile}_{label}_ramp_ident_results.npz'), allow_pickle=True)
        time_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_{n_ramp_time_shuffle}_{ramp_time_percentile}_{label}_time_cell_results.npz'), allow_pickle=True)
        cell_nums_ramp.append(ramp_ident_results['cell_nums_ramp'])
        cell_nums_time.append(time_ident_results['time_cell_nums'])
    cell_nums_ramp = np.logical_or(cell_nums_ramp[0], cell_nums_ramp[1])  # ramping cell for either stim_1 or stim_2
    cell_nums_time = np.logical_or(cell_nums_time[0], cell_nums_time[1])  # time cell for either stim_1 or stim_2


    # Load existing model
    ckpt_name = load_model_path.replace('/', '_').replace('.pt', '_pt')  # 'lstm_512_5e-06_seed_1_epi59999_pt'
    hparams = config_dir.split('_')
    hidden_type = hparams[0]
    n_neurons = int(hparams[1])
    lr = float(hparams[2])
    wd = None
    p = None
    dropout_type = None
    if len(hparams) > 3:  # weight_decay or dropout
        if 'wd' in hparams[3]:
            wd = float(hparams[3][2:])
        if 'p' in hparams[3]:
            p = float(hparams[3][1:])
            dropout_type = hparams[4]
    assert hidden_type=='lstm', 'Lesion expt must be done on LSTM network'
    env_title = 'Interval_Discrimination_2D'
    net_title = 'LSTM'
    print(hidden_type, n_neurons, lr, seed, epi, env_title, net_title)

    # Make directory in /lesion to save data and model
    agent_str = f"{seed}_{epi}_{n_ramp_time_shuffle}_{ramp_time_percentile}"
    save_dir = os.path.join(main_dir, config_dir, agent_str, argsdict['expt_type'])
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
    postlesion_nav_array = np.zeros((3, len(n_lesion), num_shuffle))
    mean_kl_div_array = np.zeros((3, len(n_lesion), num_shuffle))

    random_index_dict = generate_random_index(num_shuffle, n_neurons, cell_nums_ramp, cell_nums_time)

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


    for i_lesion_type, lesion_type in enumerate(['random', 'ramp', 'time']):
            for i_num_lesion, num_lesion in enumerate(n_lesion):
                for i_shuffle in tqdm(range(num_shuffle)):
                    gc.collect()
                    torch.cuda.empty_cache()
                    env = IntervalDiscrimination_2D(len_delay=20, len_edge=7, rwd=100, inc_rwd=-100, step_rwd=-0.1, poke_rwd=5, seed=seed)
                    if p is not None:
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
                            p_dropout=p,
                            dropout_type=dropout_type,
                            batch_size=1,
                            rfsize=rfsize,
                            padding=padding,
                            stride=stride)
                    else:
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
                    net.load_state_dict(torch.load(os.path.join('/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/agents/timing2d', load_model_path)))
                    net.eval()
                    lesion_index = random_index_dict[lesion_type][i_shuffle][:num_lesion].astype(int)
                    if argsdict['expt_type'] == 'lesion':
                        postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle], postlesion_nav_array[i_lesion_type, i_num_lesion, i_shuffle] = \
                            lesion_experiment(env=env, net=net, optimizer=optimizer,n_total_episodes=n_total_episodes,
                                              title=f"{lesion_type}_{num_lesion}",
                                              lesion_idx=lesion_index, save_dir=save_dir, save_net_and_data=save_net_and_data,
                                              backprop=backprop)
                    elif argsdict['expt_type'] == 'rehydration':
                        _, postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle], \
                        _, postlesion_nav_array[i_lesion_type, i_num_lesion, i_shuffle], \
                        mean_kl_div_array[i_lesion_type, i_num_lesion, i_shuffle] = rehydration_experiment(env=env, net=net,
                                                                                                           n_total_episodes=n_total_episodes,
                                                                                                           lesion_idx=lesion_index)

                    if verbose:
                        print(f"Lesion type: {lesion_type} ; Lesion number: {num_lesion} ; completed. {postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle]*100:.3f}% nonmatch")
                    del env, net, optimizer

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncol=1)
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
    ax1.hlines(y=0.5, xmin=0, xmax=1, transform=ax1.get_yaxis_transform(), linestyles='dashed', colors='gray')
    ax1.set_xlabel('Number of neurons lesioned')
    ax1.set_ylabel('Fraction Nonmatch')
    ax1.set_ylim(0,1)
    ax1.legend()

    ax2.plot(n_lesion, np.mean(postlesion_nav_array[0, :, :], axis=-1), color='gray', label='Random lesion')
    ax2.fill_between(n_lesion,
                     np.mean(postlesion_nav_array[0, :, :], axis=-1)-np.std(postlesion_nav_array[0, :, :], axis=-1),
                     np.mean(postlesion_nav_array[0, :, :], axis=-1)+np.std(postlesion_nav_array[0, :, :], axis=-1), color='lightgray', alpha=0.2)
    ax2.plot(n_lesion, np.mean(postlesion_nav_array[1, :, :], axis=-1), color='royalblue', label='Ramping cell lesion')
    ax2.fill_between(n_lesion,
                     np.mean(postlesion_nav_array[1, :, :], axis=-1)-np.std(postlesion_nav_array[1, :, :], axis=-1),
                     np.mean(postlesion_nav_array[1, :, :], axis=-1)+np.std(postlesion_nav_array[1, :, :], axis=-1), color='lightsteelblue', alpha=0.2)
    ax2.plot(n_lesion, np.mean(postlesion_nav_array[2, :, :], axis=-1), color='magenta', label='Time cell lesion')
    ax2.fill_between(n_lesion,
                     np.mean(postlesion_nav_array[2, :, :], axis=-1)-np.std(postlesion_nav_array[2, :, :], axis=-1),
                     np.mean(postlesion_nav_array[2, :, :], axis=-1)+np.std(postlesion_nav_array[2, :, :], axis=-1), color='thistle', alpha=0.2)
    ax2.set_xlabel('Number of neurons lesioned')
    ax2.set_ylabel('Sum of episodic navigation rewards')
    ax2.legend()
    #plt.show()
    fig.savefig(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_{argsdict["expt_type"]}_performance_results.png'))

    if argsdict["expt_type"] == "rehydration":
        fig, ax3 = plt.subplots()
        fig.suptitle(f'{env_title}_{ckpt_name}')
        ax3.plot(n_lesion, np.mean(mean_kl_div_array[0, :, :], axis=-1), color='gray', label='Random lesion')
        ax3.fill_between(n_lesion,
                         np.mean(mean_kl_div_array[0, :, :], axis=-1)-np.std(mean_kl_div_array[0, :, :], axis=-1),
                         np.mean(mean_kl_div_array[0, :, :], axis=-1)+np.std(mean_kl_div_array[0, :, :], axis=-1), color='lightgray', alpha=0.2)
        ax3.plot(n_lesion, np.mean(mean_kl_div_array[1, :, :], axis=-1), color='royalblue', label='Ramping cell lesion')
        ax3.fill_between(n_lesion,
                         np.mean(mean_kl_div_array[1, :, :], axis=-1)-np.std(mean_kl_div_array[1, :, :], axis=-1),
                         np.mean(mean_kl_div_array[1, :, :], axis=-1)+np.std(mean_kl_div_array[1, :, :], axis=-1), color='lightsteelblue', alpha=0.2)
        ax3.plot(n_lesion, np.mean(mean_kl_div_array[2, :, :], axis=-1), color='magenta', label='Time cell lesion')
        ax3.fill_between(n_lesion,
                         np.mean(mean_kl_div_array[2, :, :], axis=-1)-np.std(mean_kl_div_array[2, :, :], axis=-1),
                         np.mean(mean_kl_div_array[2, :, :], axis=-1)+np.std(mean_kl_div_array[2, :, :], axis=-1), color='thistle', alpha=0.2)
        ax3.set_xlabel('Number of neurons lesioned')
        ax3.set_ylabel('KL divergence between $\pi$ and $\pi_{new}$')
        ax3.legend()
        #plt.show()
        fig.savefig(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_{argsdict["expt_type"]}_kl_div_results.png'))

    np.savez_compressed(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_{argsdict["expt_type"]}_results.npz'),
                        random_index_dict=random_index_dict, n_lesion=n_lesion, postlesion_perf=postlesion_perf_array,
                        postlesion_nav=postlesion_nav_array, mean_kl_div=mean_kl_div_array)

