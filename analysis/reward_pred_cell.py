import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from tqdm import tqdm
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")
from utils_incentive import *
import cProfile
import pstats
import argparse

def main():
    # argparse each seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--untrained', type=bool, default=False, help='whether to use untrained model')
    parser.add_argument('--env_type', type=str, default='mem', help='mem or nomem')
    args = parser.parse_args()
    each_seed = args.seed
    untrained = args.untrained
    env_type = args.env_type

    data_dir = f"/network/scratch/l/lindongy/timecell/data_collecting/tunl2d/{env_type}_40_lstm_256_5e-06_smallrwd0.5_4"
    n_total_episodes = 5000
    len_delay = 40
    n_neurons = 256
    n_shuffle = 500
    percentile = 99
    p_small_reward = 0.5
    a_small_reward = int(data_dir[-1])
    small_rwd = 100/a_small_reward


    if untrained:
        save_dir = f'/network/scratch/l/lindongy/timecell/figures/tunl2d_incentive/{env_type}/untrained'
    else:
        save_dir = f'/network/scratch/l/lindongy/timecell/figures/tunl2d_incentive/{env_type}/trained'
    os.makedirs(save_dir, exist_ok=True)


    print(f'====================== Analyzing seed {each_seed} ...======================================')
    seed_save_dir = os.path.join(save_dir, f'seed_{each_seed}')
    if not os.path.exists(seed_save_dir):
        os.makedirs(seed_save_dir, exist_ok=True)
    # load the data
    print("load the data...")

    if untrained:
        data = np.load(os.path.join(data_dir, f'seed_{each_seed}_untrained_agent_weight_frozen_data.npz'), allow_pickle=True)
    else:
        data = np.load(os.path.join(data_dir, f'{env_type}_40_lstm_256_5e-06_smallrwd0.5_4_seed_{each_seed}_epi79999.pt_data.npz'), allow_pickle=True)

    stim = data['stim']  # (n_episodes, 2)
    choice = data['choice'] # (n_episodes, 2)
    neural_activity = data['neural_activity']  # (n_episodes,) array. Each element of the array is a list of length T_episode. Each element of the list is a numpy array of shape (n_neurons,).
    action = data['action'] # (n_episodes,) array. Each element of the array is a list of length T_episode. Each element of the list is an integer.
    location = data['location'] # (n_episodes,) array. Each element of the array is a list of length T_episode. Each element of the list is a numpy array of shape (2,).
    reward = data['reward'] # (n_episodes, ) array. Each element of the array is a list of length T_episode. Each element of the list is a float.
    epi_small_reward = data['epi_small_reward'] # (n_episodes)  # whether agent receives small reward in this episode
    epi_to_incentivize = data['epi_to_incentivize'] # (n_episodes)  # whether agent should receive small reward in this episode

    # get timestamps
    print("get timestamps...")

    timestamp_initiate_stimulus = []
    timestamp_poke_stimulus = []
    timestamp_initiate_choice = []
    timestamp_poke_choice = []  # associated with reward 5 for correct trials or -20 for incorrect trials
    timestamp_small_reward = []
    timestamp_big_reward = []  # for correct trials only; collecting final big reward
    trial_idx_small_reward = []

    for i_episode in tqdm(range(n_total_episodes)):

        action_episode = np.array(action[i_episode])  # T_episode
        location_episode = np.array(location[i_episode])  # T_episode x 2
        reward_episode = np.array(reward[i_episode])  # T_episode

        # stimulus-initiation, sample-poke, choice-initiation
        poke_reward_timestamp = np.where(np.asarray(reward_episode) == 5)[0]
        if len(poke_reward_timestamp) == 4: # correct trial
            # assert trials with 4 poke_reward_timestamps are correct trials, for mem or nomem
            if (env_type == 'mem' and np.any(stim[i_episode] != choice[i_episode])) or (env_type == 'nomem' and np.all(choice[i_episode] == [1,1])):  # good agent
                assert  reward_episode[-1] == 100
                stim_init_idx, sample_poke_idx, choice_init_idx, choice_poke_idx = poke_reward_timestamp
                timestamp_big_reward.append(len(reward_episode) - 1)  # the last item
            else:
                assert reward_episode[-1] == -20
                stim_init_idx, sample_poke_idx, choice_init_idx, _ = poke_reward_timestamp  # disregard poking the correct chice if it eventually goes to incorrect choice
                # and then, len(reward_episode) - 1 corresponds to poking incorrect choice and receiving -20 reward
                choice_poke_idx = len(reward_episode) - 1  # the last item
        elif len(poke_reward_timestamp) == 3: # incorrect trial
            # assert trials with 3 poke_reward_timestamps are incorrect trials, for mem or nomem
            if env_type == 'mem':
                assert np.all(stim[i_episode] == choice[i_episode])
            elif env_type == 'nomem':
                assert np.all(choice[i_episode] == [1,7])
            assert reward_episode[-1] == -20

            stim_init_idx, sample_poke_idx, choice_init_idx = poke_reward_timestamp
            choice_poke_idx = len(reward_episode) - 1  # the last item
        else:
            raise ValueError("Incorrect number of timestamps for reward: either 3 for incorrect trials or 4 for correct trials.")
        timestamp_initiate_stimulus.append(stim_init_idx)
        timestamp_initiate_choice.append(choice_init_idx)
        timestamp_poke_stimulus.append(sample_poke_idx)
        timestamp_poke_choice.append(choice_poke_idx)
        if len(np.where(reward_episode==small_rwd)[0]) > 0:
            timestamp_small_reward.append(np.where(reward_episode==small_rwd)[0][0])
            trial_idx_small_reward.append(i_episode)


    timestamp_initiate_stimulus = np.asarray(timestamp_initiate_stimulus)
    timestamp_initiate_choice = np.asarray(timestamp_initiate_choice)
    timestamp_small_reward = np.asarray(timestamp_small_reward)
    timestamp_big_reward = np.asarray(timestamp_big_reward)
    timestamp_poke_stimulus = np.asarray(timestamp_poke_stimulus)
    timestamp_poke_choice = np.asarray(timestamp_poke_choice)

    trial_idx_initate_stimulus = trial_idx_intiate_choice = np.arange(5000)

    # get correct-choice and incorrect-choice trial indices
    trial_idx_nonmatch = np.where(np.any(stim != choice, axis=1))[0]  # correct for mem
    trial_idx_match = np.where(np.all(stim == choice, axis=1))[0]  # incorrect for mem
    trial_idx_left_choice = np.where(np.all(choice==[1,1], axis=1))[0] # correct for nomem
    trial_idx_right_choice = np.where(np.all(choice==[1,7], axis=1))[0] # incorrect for nomem
    trial_idx_left_stimulus = np.where(np.all(stim==[1,1], axis=1))[0]
    trial_idx_right_stimulus = np.where(np.all(stim==[1,7], axis=1))[0]

    if env_type=="mem":
        assert len(trial_idx_nonmatch) == len(timestamp_big_reward)
    else:
        assert len(trial_idx_left_choice) == len(timestamp_big_reward)

    max_trial_length = max(len(trial) for trial in neural_activity)
    n_neurons = neural_activity[0][0].shape[0]
    n_trials = len(neural_activity)

    neural_activity_padded = np.full((n_trials, n_neurons, max_trial_length), np.nan)

    for i_trial, trial in enumerate(neural_activity):
        for t, data in enumerate(trial):
            neural_activity_padded[i_trial, :, t] = data

    def convert_to_float16(neural_activity):
        # Assuming neural_activity is a list of lists of numpy arrays
        for i_episode, episode in enumerate(neural_activity):
            for i_time, data in enumerate(episode):
                # Convert each numpy array to float16
                neural_activity[i_episode][i_time] = data.astype(np.float16)
        return neural_activity

    # Now call this function with your neural_activity data
    neural_activity = convert_to_float16(neural_activity)

    neural_activity_padded = np.float16(neural_activity_padded)


    # Note: trial_idx should match timestamp length-wise

    # identify neurons that fire significantly at different time steps
    print("identify neurons that fire significantly when collecting small reward...")
    small_reward_cells_idx = identify_significant_neurons(neural_activity, neural_activity_padded, trial_idx=trial_idx_small_reward, timestamp=timestamp_small_reward, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'small_reward_cells_window'))
    print(f"Number of neurons that fire significantly when collecting small reward: {len(small_reward_cells_idx)}")

    print("identify neurons that fire significantly when initiating stimulus phase...")
    initiate_stimulus_cells_idx = identify_significant_neurons(neural_activity, neural_activity_padded, trial_idx=trial_idx_initate_stimulus, timestamp=timestamp_initiate_stimulus, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'init_stim_cells_window'))
    print(f"Number of neurons that fire significantly when initiating stimulus phase: {len(initiate_stimulus_cells_idx)}")

    print("identify neurons that fire significantly when initiating choice phase...")
    initiate_choice_cells_idx = identify_significant_neurons(neural_activity, neural_activity_padded, trial_idx=trial_idx_intiate_choice, timestamp=timestamp_initiate_choice, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'init_choice_cells_window'))
    print(f"Number of neurons that fire significantly when initiating choice phase: {len(initiate_choice_cells_idx)}")

    print("identify neurons that fire significantly when collecting big reward...")
    big_reward_cells_idx = identify_significant_neurons(neural_activity, neural_activity_padded, trial_idx=trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp=timestamp_big_reward, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'big_reward_cells_window'))
    print(f"Number of neurons that fire significantly when collecting big reward: {len(big_reward_cells_idx)}")

    print("identify neurons that fire significantly between poking stimulus and collecting incentive..")
    incentive_prediction_cells_idx = identify_significant_neurons_between_timestamps(neural_activity_padded, trial_idx=trial_idx_small_reward, timestamp_start=timestamp_poke_stimulus[trial_idx_small_reward], timestamp_end=timestamp_small_reward, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'stim_to_incentive_cells'))
    print(f"Number of neurons that fire significantly between poking stimulus and collecting incentive: {len(incentive_prediction_cells_idx)}")

    print("identify neurons that fire significantly between poking choice and collecting big reward..")
    reward_prediction_cells_idx = identify_significant_neurons_between_timestamps(neural_activity_padded, trial_idx=trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp_start=timestamp_poke_choice[trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice], timestamp_end=timestamp_big_reward, percentile=percentile, n_shuffle=n_shuffle, plot=True, save_dir=os.path.join(seed_save_dir, 'choice_to_reward_cells'))
    print(f"Number of neurons that fire significantly between poking choice and collecting big reward: {len(reward_prediction_cells_idx)}")

    # Save the idx of identified cells to one file
    np.savez(os.path.join(seed_save_dir, 'identified_cells_idx.npz'), small_reward_cells_idx=small_reward_cells_idx, initiate_stimulus_cells_idx=initiate_stimulus_cells_idx, initiate_choice_cells_idx=initiate_choice_cells_idx, big_reward_cells_idx=big_reward_cells_idx, incentive_prediction_cells_idx=incentive_prediction_cells_idx, reward_prediction_cells_idx=reward_prediction_cells_idx)

    # for small_reward_cells, plot aggregated activity around small reward timestamp
    print("plot aggregated activity around small reward timestamp...")
    aggregated_activity = aggregate_neural_activity(neural_activity, trial_idx_small_reward, timestamp_small_reward, window_size=5, n_neurons=256)
    plot_aggregated_activity(aggregated_activity, small_reward_cells_idx, save_dir=os.path.join(seed_save_dir, 'small_reward', 'first_100'), window_size=5, align=None, n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity, small_reward_cells_idx, save_dir=os.path.join(seed_save_dir, 'small_reward', 'random'), window_size=5, align=None, n_trials=100, random_trials=True)

    # for big_reward_cells, plot aggregated activity around big reward timestamp
    print("plot aggregated activity around big reward timestamp...")
    aggregated_activity = aggregate_neural_activity(neural_activity, trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp_big_reward, window_size=5, n_neurons=256)
    plot_aggregated_activity(aggregated_activity, big_reward_cells_idx, save_dir=os.path.join(seed_save_dir, 'big_reward', 'first_100'), window_size=5, align=None, n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity, big_reward_cells_idx, save_dir=os.path.join(seed_save_dir, 'big_reward', 'random'), window_size=5, align=None, n_trials=100, random_trials=True)

    # for incentive_prediction_cells, plot aggregated activity between poking stimulus and collecting incentive
    print("plot aggregated activity between poking stimulus and collecting incentive...")
    aggregated_activity_start_aligned, aggregated_activity_end_aligned = aggregate_run_to_reward_activity(neural_activity, trial_idx_small_reward, timestamp_poke_stimulus[trial_idx_small_reward], timestamp_small_reward, n_neurons=256)
    plot_aggregated_activity(aggregated_activity_start_aligned, incentive_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'incentive_prediction_start_aligned', 'first_100'), window_size=None, align='start', n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity_start_aligned, incentive_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'incentive_prediction_start_aligned', 'random'), window_size=None, align='start', n_trials=100, random_trials=True)
    plot_aggregated_activity(aggregated_activity_end_aligned, incentive_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'incentive_prediction_end_aligned', 'first_100'), window_size=None, align='end', n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity_end_aligned, incentive_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'incentive_prediction_end_aligned', 'random'), window_size=None, align='end', n_trials=100, random_trials=True)

    # for reward_prediction_cells, plot aggregated activity between poking choice and collecting big reward
    print("plot aggregated activity between poking choice and collecting big reward...")
    aggregated_activity_start_aligned, aggregated_activity_end_aligned = aggregate_run_to_reward_activity(neural_activity, trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp_poke_choice[trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice], timestamp_big_reward, n_neurons=256)
    plot_aggregated_activity(aggregated_activity_start_aligned, reward_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'reward_prediction_start_aligned', 'first_100'), window_size=None, align='start', n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity_start_aligned, reward_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'reward_prediction_start_aligned', 'random'), window_size=None, align='start', n_trials=100, random_trials=True)
    plot_aggregated_activity(aggregated_activity_end_aligned, reward_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'reward_prediction_end_aligned', 'first_100'), window_size=None, align='end', n_trials=100, random_trials=False)
    plot_aggregated_activity(aggregated_activity_end_aligned, reward_prediction_cells_idx, save_dir=os.path.join(seed_save_dir, 'reward_prediction_end_aligned', 'random'), window_size=None, align='end', n_trials=100, random_trials=True)

    # plot overlapping tuning curves for stimulus-incentive run and choice-reward run
    print("plot overlapping tuning curves for stimulus-incentive run and choice-reward run...")
    aggregated_activity_start_aligned_small, aggregated_activity_end_aligned_small = aggregate_run_to_reward_activity(neural_activity, trial_idx_small_reward, timestamp_poke_stimulus[trial_idx_small_reward], timestamp_small_reward, n_neurons=256)
    aggregated_activity_start_aligned_big, aggregated_activity_end_aligned_big = aggregate_run_to_reward_activity(neural_activity, trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp_poke_choice[trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice], timestamp_big_reward, n_neurons=256)
    plot_overlap_tuning_curves_for_two_aggregated_activity(aggregated_activity_start_aligned_small, aggregated_activity_start_aligned_big, 'stim-incentive', 'choice-reward', neuron_idx=np.arange(256), save_dir=os.path.join(seed_save_dir, 'reward_pred_comparison', 'start_aligned'), align='start', window_size=None)
    plot_overlap_tuning_curves_for_two_aggregated_activity(aggregated_activity_start_aligned_small, aggregated_activity_start_aligned_big, 'stim-incentive', 'choice-reward', neuron_idx=np.arange(256), save_dir=os.path.join(seed_save_dir, 'reward_pred_comparison', 'end_aligned'), align='end', window_size=None)

    # plot overlapping tuning curves for small_reward and big_reward
    print("plot overlapping tuning curves for small_reward and big_reward...")
    aggregated_activity_small_reward = aggregate_neural_activity(neural_activity, trial_idx_small_reward, timestamp_small_reward, window_size=5, n_neurons=256)
    aggregated_activity_big_reward = aggregate_neural_activity(neural_activity, trial_idx_nonmatch if env_type=="mem" else trial_idx_left_choice, timestamp_big_reward, window_size=5, n_neurons=256)
    plot_overlap_tuning_curves_for_two_aggregated_activity(aggregated_activity_small_reward, aggregated_activity_big_reward, 'small_reward', 'big_reward', neuron_idx=np.arange(256), save_dir=os.path.join(seed_save_dir, 'reward_comparison'), align=None, window_size=5)


if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()

    main()

    profile.disable()
    stats = pstats.Stats(profile).sort_stats('cumulative')
    stats.print_stats()

