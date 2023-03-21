import random
import os
from agents.model_2d import *
from envs.tunl_2d import Tunl
import numpy as np
import torch
from lesion_expt_utils import generate_random_index
import argparse
from tqdm import tqdm
import re
import gc
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
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
        if not os.path.exists(net_and_data_dir):
            os.mkdir(net_and_data_dir)
        #torch.save(net.state_dict(), os.path.join(net_and_data_dir, f'postlesion_{title}.pt'))
        np.savez_compressed(os.path.join(net_and_data_dir, f'lesion_{title}_data.npz'), stim=stim, choice=choice, ct=ct, delay_loc=delay_loc,
                            delay_resp_hx=delay_resp_hx,
                            delay_resp_cx=delay_resp_cx,
                            epi_nav_reward=epi_nav_reward,
                            ideal_nav_rwds=ideal_nav_rwds)
    return np.mean(nonmatch_perc), np.mean(epi_nav_reward)


def rehydration_experiment(env, net, n_total_episodes, lesion_idx):
    # new policy is calculated from silencing lesion_idx WHILE other hx stay the same

    ct = np.zeros(n_total_episodes, dtype=np.int8)  # whether it's a correction trial or not
    stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
    ideal_nav_rwds = np.zeros(n_total_episodes, dtype=np.float16)

    epi_nav_reward = np.zeros(n_total_episodes, dtype=np.float16)
    nonmatch_perc = np.zeros(n_total_episodes, dtype=np.float16)
    choice = np.zeros((n_total_episodes, 2), dtype=np.int8)  # record the location when done

    epi_nav_reward_prime = np.zeros(n_total_episodes, dtype=np.float16)
    nonmatch_perc_prime = np.zeros(n_total_episodes, dtype=np.float16)

    kl_div = []

    for i_episode in tqdm(range(n_total_episodes)):  # one episode = one sample
        done = False
        env.reset()
        ideal_nav_rwds[i_episode] = ideal_nav_rwd(env=env, len_edge=7, len_delay=env.len_delay, step_rwd=-0.1, poke_rwd=5)
        net.reinit_hid()
        stim[i_episode] = env.sample_loc
        ct[i_episode] = int(env.correction_trial)

        pi_record = []
        pi_prime_record = []

        action_record = []
        action_prime_record = []

        nav_reward_prime = 0
        while not done:
            obs = torch.unsqueeze(torch.Tensor(np.reshape(env.observation, (3, env.h, env.w))), dim=0).float().to(device)
            pol, val = net.forward(obs)
            # pol, val, lin_act = net.forward(torch.as_tensor(env.observation).float().to(device))  # here, pol is original policy
            pi_record.append(pol)
            new_activity = net.hx[net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            new_activity[lesion_idx] = 0  # the new, manipulated hx
            lin_out = F.relu(net.hidden[net.hidden_types.index("linear")](torch.from_numpy(new_activity).to(device)))
            new_pol = F.softmax(net.output[0](lin_out), dim=0)
            pi_prime_record.append(new_pol)

            # select action from old and new policy
            action = Categorical(pol).sample().item()
            action_record.append(action)
            new_action = Categorical(new_pol).sample().item()
            action_prime_record.append(new_action)

            # proceed with trial with old policy
            new_obs, reward, done = env.step(action)
            # calculate hypothetical reward if the new action from rehydrated network was taken
            reward_prime = env.calc_reward_without_stepping(new_action)
            if reward_prime == env.step_rwd or reward_prime == env.poke_rwd or reward_prime == 0:  # navigation reward
                nav_reward_prime += reward_prime

        choice[i_episode] = env.current_loc
        if np.any(stim[i_episode] != choice[i_episode]):  # nonmatch
            nonmatch_perc[i_episode] = 1
            breakpoint()  # check: reward == env.rwd
        if reward_prime == env.rwd:
            nonmatch_perc_prime[i_episode] = 1
        epi_nav_reward[i_episode] = env.nav_reward
        epi_nav_reward_prime[i_episode] = nav_reward_prime

        # calculate KL divergence between original policy pi and new policy pi'
        kl_divergence = 0
        for pi, pi_p in zip(pi_record, pi_prime_record):
            kl_divergence += F.kl_div(pi_p, pi, reduction='batchmean')
        kl_divergence /= len(pi_record)
        kl_div.append(kl_divergence.item())
        # print('KL divergence:', kl_divergence.item())

    del stim, ct, ideal_nav_rwds, choice

    return np.mean(nonmatch_perc), np.mean(nonmatch_perc_prime), np.mean(epi_nav_reward), np.mean(epi_nav_reward_prime), np.mean(np.asarray(kl_div))


if __name__ == '__main__':

    utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

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
    if not os.path.exists(os.path.join('/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/agents/tunl2d', load_model_path)):
        print("agent does not exist. exiting.")
    sys.exit()

    # Load_model_path: mem_40_lstm_256_5e-06/seed_1_epi999999.pt, relative to training/tunl2d
    config_dir = load_model_path.split('/')[0]
    pt_name = load_model_path.split('/')[1]
    pt = re.match("seed_(\d+)_epi(\d+).pt", pt_name)
    seed = int(pt[1])
    epi = int(pt[2])
    main_dir = '/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/analysis_results/tunl2d'
    data_analysis_dir = os.path.join(main_dir, config_dir)
    ramp_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_{n_ramp_time_shuffle}_{ramp_time_percentile}_ramp_ident_results.npz'), allow_pickle=True)
    time_ident_results = np.load(os.path.join(data_analysis_dir,f'{seed}_{epi}_{n_ramp_time_shuffle}_{ramp_time_percentile}_time_cell_results.npz'), allow_pickle=True)
    cell_nums_ramp = ramp_ident_results['cell_nums_ramp']
    cell_nums_time = time_ident_results['time_cell_nums']

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
                net.load_state_dict(torch.load(os.path.join('/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/agents/tunl2d', load_model_path)))
                net.eval()

                lesion_index = random_index_dict[lesion_type][i_shuffle][:num_lesion].astype(int)

                if argsdict['expt_type'] == 'lesion':
                    postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle], postlesion_nav_array[i_lesion_type, i_num_lesion, i_shuffle] = \
                        lesion_experiment(env=env, net=net, optimizer=optimizer,n_total_episodes=n_total_episodes,
                                          title=f"{lesion_type}_{num_lesion}",
                                          lesion_idx=lesion_index, save_dir=save_dir, save_net_and_data=save_net_and_data,
                                          backprop=backprop)
                elif argsdict['expt_type'] == 'rehydration':
                    _, postlesion_perf_array[i_lesion_type, i_num_lesion, i_shuffle],\
                    _, postlesion_nav_array[i_lesion_type, i_num_lesion, i_shuffle],\
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
        ax3.set_ylabel('KL divergence between $\pi$ and $\pi_new$')
        ax3.legend()
        #plt.show()
        fig.savefig(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_{argsdict["expt_type"]}_kl_div_results.png'))

    np.savez_compressed(os.path.join(save_dir, f'epi{n_total_episodes}_shuff{num_shuffle}_idx{lesion_idx_start}_{lesion_idx_step}_{n_neurons if lesion_idx_end is None else lesion_idx_end}_{argsdict["expt_type"]}_results.npz'),
                        random_index_dict=random_index_dict, n_lesion=n_lesion, postlesion_perf=postlesion_perf_array,
                        postlesion_nav=postlesion_nav_array, mean_kl_div=mean_kl_div_array)

