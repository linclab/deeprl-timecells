import random
import os
from re import I
from expts.agents.model_1d import *
from expts.envs.int_discrim import IntervalDiscrimination
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


# Train and record
# Initialize arrays for recording
def lesion_experiment(env, net, optimizer, n_total_episodes, lesion_idx, title, save_dir, backprop=False):
    action_hist = np.zeros(n_total_episodes, dtype=np.int8)
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
    stim = np.zeros((n_total_episodes, 2), dtype=np.int8)
    stim1_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    stim2_resp = np.zeros((n_total_episodes, 40, n_neurons), dtype=np.float32)
    delay_resp = np.zeros((n_total_episodes, 20, n_neurons), dtype=np.float32)

    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        env.reset()
        net.reinit_hid()
        stim[i_episode,0] = env.first_stim
        stim[i_episode,1] = env.second_stim
        while not done:
            pol, val, lin_act = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0), lesion_idx=lesion_idx)  # forward
            if env.task_stage in ['init', 'choice_init']:
                act, p, v = select_action(net, pol, val)
                new_obs, reward, done = env.step(act)
                net.rewards.append(reward)
            else:
                new_obs, reward, done = env.step()

            if env.task_stage == 'first_stim' and env.elapsed_t > 0:
                stim1_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'second_stim' and env.elapsed_t > 0:
                stim2_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()
            elif env.task_stage == 'delay' and env.elapsed_t > 0:
                delay_resp[i_episode, env.elapsed_t-1, :] = net.hx[
                    net.hidden_types.index("lstm")].clone().detach().cpu().numpy().squeeze()

            if env.task_stage == 'choice_init':
                action_hist[i_episode] = act
                correct_trial[i_episode] = env.correct_trial
        if backprop:
            p_loss, v_loss = finish_trial(net, 1, optimizer)
    torch.save(net.state_dict(), save_dir + f'/postlesion_{title}.pt')

    np.savez_compressed(save_dir + f'/lesion_{title}_data.npz', action_hist=action_hist, correct_trial=correct_trial,
                            stim=stim, stim1_resp_hx=stim1_resp,
                            stim2_resp_hx=stim2_resp, delay_resp_hx=delay_resp)
    return np.mean(correct_trial)


if __name__ == '__main__':

    plot_utils.linclab_plt_defaults()
    plot_utils.set_font(font='Helvetica')

    parser = argparse.ArgumentParser(description="lesion study in Head-fixed Interval Discrimination task simulation")
    parser.add_argument("--n_total_episodes",type=int,default=1000,help="Total episodes to run lesion expt")
    parser.add_argument("--load_model_path", type=str, default='None', help="path RELATIVE TO $SCRATCH/timecell/training/timing")
    parser.add_argument("--backprop", type=bool, default=False, help="Whether backprop loss during lesion expt")
    args = parser.parse_args()
    argsdict = args.__dict__
    print(argsdict)

    n_total_episodes = argsdict['n_total_episodes']
    load_model_path = argsdict['load_model_path']
    backprop = True if argsdict['backprop'] == True or argsdict['backprop'] == 'True' else False

    # Load_model_path: lstm_256_5e-06/seed_1_epi149999.pt, relative to training/timing
    config_dir = load_model_path.split('/')[0]
    data_name = load_model_path.replace('/', '_')+'_data.npz'
    data = np.load(os.path.join('/network/scratch/l/lindongy/timecell/data_collecting/timing', config_dir, data_name), allow_pickle=True)

    # Load existing model
    ckpt_name = load_model_path.replace('/', '_').replace('.pt', '_pt')  # 'lstm_512_5e-06_seed_1_epi59999_pt'
    hparams = ckpt_name.split('_')
    hidden_type = hparams[0]
    n_neurons = int(hparams[1])
    lr = float(hparams[2])
    seed = int(hparams[4])
    epi = int(hparams[5][3:])
    assert hidden_type=='lstm', 'Lesion expt must be done on LSTM network'
    env_title = 'Interval_Discrimination'
    net_title = 'LSTM'
    print(hidden_type, n_neurons, lr, seed, epi, env_title, net_title)


    stim = data["stim"]
    n_data_episodes = np.shape(stim)[0]
    print(f"Total number of episodes in saved data: {n_data_episodes}")
    if n_data_episodes > 5000:
        stim = stim[:-5000]
        action_hist = data["action_hist"][:-5000]
        correct_trials = data["correct_trial"][:-5000]
        stim1_resp = data["stim1_resp_hx"][:-5000]
        stim2_resp = data["stim2_resp_hx"][:-5000]
        delay_resp = data["delay_resp_hx"][:-5000]
    else:
        action_hist = data["action_hist"]
        correct_trials = data["correct_trial"]
        stim1_resp = data["stim1_resp_hx"]
        stim2_resp = data["stim2_resp_hx"]
        delay_resp = data["delay_resp_hx"]

    # TODO: write a function to analyse cell_nums_ramp, cell_nums_seq, cell_nums_nontime

    stim_set = np.unique(stim)
    num_stim = np.max(np.shape(stim_set))

    # Make directory in /lesion to save data and model
    main_dir = '/network/scratch/l/lindongy/timecell/lesion/timing'
    save_dir = os.path.join(main_dir, f'{hidden_type}_{n_neurons}_{lr}_seed{seed}_epi{epi}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(f'Saving to {save_dir}')

    # Setting up cuda and seeds
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")  # Not using cuda for 1D IntDiscrm
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = IntervalDiscrimination(seed=seed)
    net = AC_Net(
        input_dimensions=2,  # input dim
        action_dimensions=2,  # action dim
        hidden_types=[hidden_type, 'linear'],  # hidden types
        hidden_dimensions=[n_neurons, n_neurons],  # hidden dims
        batch_size=1)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    n_lesion = np.arange(start=0, stop=n_neurons, step=10)
    postlesion_perf_array = np.zeros((3, len(n_lesion)))

    for i_row, lesion_type in enumerate(['random', 'ramp', 'seq']):
        for i_column, num_lesion in enumerate(n_lesion):

            net.load_state_dict(torch.load(os.path.join('/network/scratch/l/lindongy/timecell/training/timing', load_model_path), map_location=torch.device('cpu')))
            net.eval()

            lesion_index = generate_lesion_index(lesion_type, num_lesion, n_neurons=n_neurons, cell_nums_ramp=cell_nums_ramp, cell_nums_seq=cell_nums_seq)

            postlesion_perf_array[i_row, i_column] = lesion_experiment(env=env, net=net,optimizer=optimizer,
                                                                       n_total_episodes=n_total_episodes,
                                                                       lesion_idx=lesion_index,
                                                                       title=f"{lesion_type}_{num_lesion}",
                                                                       save_dir=save_dir,
                                                                       backprop=backprop)
            print("Lesion type:", lesion_type, "; Lesion number:", num_lesion, "completed.")

    np.savez_compressed(os.path.join(save_dir, 'lesion_performance_arr.npz'), postlesion_perf=postlesion_perf_array)