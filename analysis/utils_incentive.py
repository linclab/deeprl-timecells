import numpy as np
from scipy import stats
import itertools
from utils_mutual_info import shuffle_activity
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import sklearn
import scikit_posthocs as sp
import utils_linclab_plot
from utils_analysis import shuffle_activity, shuffle_activity_single_neuron, shuffle_activity_single_neuron_varying_duration
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

#===================================================================================================
# trial_neural_activity = neural_activity[i_episode].T  # n_neurons x T_episode
def circular_shuffle_neural_activity(trial_neural_activity):
    # trial_neural_activity: n_neurons x T_episode
    # circularly shift bins for each neuron within a trial (i.e., row-wise), thus destroying
    # pairwise correlations but preserving single-cell autocorrelations.
    # return: n_neurons x T_episode
    n_neurons = trial_neural_activity.shape[0]
    T_episode = trial_neural_activity.shape[1]
    trial_neural_activity_shuffled = np.zeros((n_neurons, T_episode))
    for i_neuron in range(n_neurons):
        trial_neural_activity_shuffled[i_neuron] = np.roll(trial_neural_activity[i_neuron], np.random.randint(T_episode))
    return trial_neural_activity_shuffled  # n_neurons x T_episode

def swap_shuffle_neural_activity(trial_neural_activity):
    # trial_neural_activity: n_neurons x T_episode
    # randomly permute columns of the trial_neural_activity,
    # thus preserving instantaneous pairwise correlations but destroying autocorrelations
    # return: n_neurons x T_episode
    n_neurons = trial_neural_activity.shape[0]
    T_episode = trial_neural_activity.shape[1]
    trial_neural_activity_shuffled = np.zeros((n_neurons, T_episode))
    for i_neuron in range(n_neurons):
        trial_neural_activity_shuffled[i_neuron] = trial_neural_activity[i_neuron][np.random.permutation(T_episode)]
    return trial_neural_activity_shuffled

#===================================================================================================

# identify if neuron's average activity at timestamp across trials is significantly (>99 percentile) higher than circularly shuffled average activity of this neuron at this timestamp across trials
def get_trial_average_value_at_time_step(neural_activity, trial_idx, timestamp, n_neurons=256):
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    trial_average_value = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        trial_average_value[i_neuron] = np.mean([neural_activity[trial_idx[i_trial]][timestamp[i_trial]][i_neuron] for i_trial in range(n_trials)])
    return trial_average_value  # n_neurons


def get_trial_average_value_within_window(neural_activity, trial_idx, timestamp, window_size=5, n_neurons=256):
    # get trial average value within window_size bins of timestamp
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    trial_average_value = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        trial_average_value[i_neuron] = np.mean([np.mean(neural_activity[trial_idx[i_trial]][timestamp[i_trial]-window_size:timestamp[i_trial]+window_size+1][i_neuron]) for i_trial in range(n_trials)])
    return trial_average_value  # n_neurons

def get_trial_average_value_between_timestamps(neural_activity, trial_idx, timestamp_start, timestamp_end, n_neurons=256):
# get trial average value between timestamp_start and timestamp_end
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp_start) == len(timestamp_end)
    trial_average_value = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        trial_average_value[i_neuron] = np.mean([np.mean(neural_activity[trial_idx[i_trial]][timestamp_start[i_trial]:timestamp_end[i_trial]][i_neuron]) for i_trial in range(n_trials)])
    return trial_average_value  # n_neurons

def get_trial_average_value_at_time_step_shuffled(neural_activity, trial_idx, timestamp, n_neurons=256, shuffle='circular'):
    # use circular shuffle or swap shuffle to shuffle neural activity
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    trial_average_value_shuffled = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        if shuffle == 'circular':
            trial_average_value_shuffled[i_neuron] = np.mean([circular_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp[i_trial]] for i_trial in range(n_trials)])
        elif shuffle == 'swap':
            trial_average_value_shuffled[i_neuron] = np.mean([swap_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp[i_trial]] for i_trial in range(n_trials)])
    return trial_average_value_shuffled  # n_neurons

def get_trial_average_value_within_window_shuffled(neural_activity, trial_idx, timestamp, window_size=5, n_neurons=256, shuffle='circular'):
    # shuffle neural activity, then take average within window_size bins of timestamp
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    trial_average_value_shuffled = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        if shuffle == 'circular':
            trial_average_value_shuffled[i_neuron] = np.mean([np.mean(circular_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp[i_trial]-window_size:timestamp[i_trial]+window_size+1]) for i_trial in range(n_trials)])
        elif shuffle == 'swap':
            trial_average_value_shuffled[i_neuron] = np.mean([np.mean(swap_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp[i_trial]-window_size:timestamp[i_trial]+window_size+1]) for i_trial in range(n_trials)])

def get_trial_average_value_between_timestamps_shuffled(neural_activity, trial_idx, timestamp_start, timestamp_end, n_neurons=256, shuffle='circular'):
    # shuffle neural activity, then take average between timestamp_start and timestamp_end
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp_start) == len(timestamp_end)
    trial_average_value_shuffled = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in range(n_neurons):
        if shuffle == 'circular':
            trial_average_value_shuffled[i_neuron] = np.mean([np.mean(circular_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp_start[i_trial]:timestamp_end[i_trial]]) for i_trial in range(n_trials)])
        elif shuffle == 'swap':
            trial_average_value_shuffled[i_neuron] = np.mean([np.mean(swap_shuffle_neural_activity(np.asarray(neural_activity[trial_idx[i_trial]]).T)[i_neuron][timestamp_start[i_trial]:timestamp_end[i_trial]]) for i_trial in range(n_trials)])
    return trial_average_value_shuffled  # n_neurons


def identify_significant_neurons(neural_activity, trial_idx, timestamp, use_window_average=False, window_size=5, n_shuffle=1000, n_neurons=256, percentile=99, shuffle='circular'):
    # identify if neuron's average activity at timestamp across trials is significantly (>99 percentile) higher than 1000 circularly shuffled average activities of this neuron at this timestamp across trials
    if use_window_average:
        trial_average_value = get_trial_average_value_within_window(neural_activity, trial_idx, timestamp, window_size=window_size, n_neurons=n_neurons)
    else:
        trial_average_value = get_trial_average_value_at_time_step(neural_activity, trial_idx, timestamp, n_neurons=n_neurons)
    trial_average_value_shuffled = []
    for i_shuffle in tqdm(range(n_shuffle)):
        if use_window_average:
            trial_average_value_shuffled.append(get_trial_average_value_within_window_shuffled(neural_activity, trial_idx, timestamp, window_size=window_size, n_neurons=n_neurons, shuffle=shuffle))
        else:
            trial_average_value_shuffled.append(get_trial_average_value_at_time_step_shuffled(neural_activity, trial_idx, timestamp, n_neurons=n_neurons, shuffle=shuffle))
    return np.where(trial_average_value > np.percentile(trial_average_value_shuffled, percentile))[0] # indices of neurons that are significantly higher than shuffled neurons

def identify_significant_neurons_between_timestamps(neural_activity, trial_idx, timestamp_start, timestamp_end, n_shuffle=1000, n_neurons=256, percentile=99, shuffle='circular'):
    trial_average_value = get_trial_average_value_between_timestamps(neural_activity, trial_idx, timestamp_start, timestamp_end, n_neurons=n_neurons)
    trial_average_value_shuffled = []
    for i_shuffle in tqdm(range(n_shuffle)):
        trial_average_value_shuffled.append(get_trial_average_value_between_timestamps_shuffled(neural_activity, trial_idx, timestamp_start, timestamp_end, n_neurons=n_neurons, shuffle=shuffle))
    return np.where(trial_average_value > np.percentile(trial_average_value_shuffled, percentile))[0] # indices of neurons that are significantly higher than shuffled neurons




# aggregate neural activity across trials for timestamp +/- 5 bins, given neuron number, trial indices, and timestamps
# if timestamp +5 bins is out of length of trial, then fill with nan
def aggregate_neural_activity(neural_activity, trial_idx, timestamp, window_size=5, n_neurons=256):
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    aggregated_activity = np.zeros((n_trials, window_size*2+1, n_neurons))
    for i_neuron in range(n_neurons):
        agg_activity = np.empty((n_trials, window_size*2+1))
        agg_activity[:] = np.nan
        for i_trial in range(n_trials):
            trial_neural_activity = np.asarray(neural_activity[trial_idx[i_trial]]).T # n_neurons x T_episode
            timestamp_trial = timestamp[i_trial]
            if timestamp_trial - window_size < 0:  # eg. initate stimulus
                agg_activity[i_trial][window_size-timestamp_trial:] = trial_neural_activity[i_neuron][:timestamp_trial+window_size+1]
            elif timestamp_trial + window_size > len(trial_neural_activity[i_neuron]):  # eg. poke choice
                agg_activity[i_trial][:len(trial_neural_activity[i_neuron])-timestamp_trial+window_size+1] = trial_neural_activity[i_neuron][timestamp_trial-window_size-1:]
            else:
                agg_activity[i_trial] = trial_neural_activity[i_neuron][timestamp_trial-window_size:timestamp_trial+window_size+1]
        aggregated_activity[:,:,i_neuron] = agg_activity
    return aggregated_activity  # n_trials x window_size*2+1 x n_neurons


def aggregate_run_to_reward_activity(neural_activity, trial_idx, timestamp_start, timestamp_end, n_neurons=256):
    # trial_idx is all 5000 for sample-smallreward run, or all trial_idx_nonmatch or trial_idx_left_choice for choice-bigreward run
    n_trials = len(timestamp_start)
    assert n_trials == len(timestamp_end) == len(trial_idx)
    aggregated_activity_start_aligned = np.zeros((n_trials, max(timestamp_end-timestamp_start), n_neurons))
    aggregated_activity_end_aligned = np.zeros((n_trials, max(timestamp_end-timestamp_start), n_neurons))
    for i_neuron in range(n_neurons):
        agg_activity_start_aligned = np.empty((n_trials, max(timestamp_end-timestamp_start)))
        agg_activity_start_aligned[:] = np.nan
        agg_activity_end_aligned = np.empty((n_trials, max(timestamp_end-timestamp_start)))
        agg_activity_end_aligned[:] = np.nan
        for i_trial in range(n_trials):
            trial_neural_activity = np.asarray(neural_activity[trial_idx[i_trial]]).T # n_neurons x T_episode
            length = timestamp_end[i_trial] - timestamp_start[i_trial]
            agg_activity_start_aligned[i_trial][:length] = trial_neural_activity[i_neuron][timestamp_start[i_trial]:timestamp_end[i_trial]]
            agg_activity_end_aligned[i_trial][-length:] = trial_neural_activity[i_neuron][timestamp_start[i_trial]:timestamp_end[i_trial]]
        aggregated_activity_start_aligned[:,:,i_neuron] = agg_activity_start_aligned
        aggregated_activity_end_aligned[:,:,i_neuron] = agg_activity_end_aligned
    return aggregated_activity_start_aligned, aggregated_activity_end_aligned  # n_trials x max(timestamp_end-timestamp_start) x n_neurons

# Plot each neurons's aggregated activity
def plot_aggregated_activity(aggregated_activity, neuron_idx, save_dir, window_size=5, align=None, n_trials=100,
                             random_trials=False):
    for i_neuron in neuron_idx:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 2]})

        # Plot heatmap
        if len(aggregated_activity) < n_trials:
            n_trials = len(aggregated_activity)
            trials_to_plot = np.arange(n_trials)
        else:
            trial_idx = np.arange(len(aggregated_activity))
            if random_trials:
                trials_to_plot = np.sort(np.random.choice(trial_idx, size=n_trials, replace=False))
            else:
                trials_to_plot = trial_idx[:n_trials]  # plot the first n_trials trials

        if align is not None:
            assert align == 'start' or align == 'end'
            nan_counts = np.sum(np.isnan(aggregated_activity[trials_to_plot, :, i_neuron]), axis=1)
            # The number of floats in each row is the total number of columns minus the number of NaNs
            float_counts = aggregated_activity[trials_to_plot, :, i_neuron].shape[1] - nan_counts
            # Find the maximum count of floats in a row
            length_to_plot = np.max(float_counts)

        if align=='start':
            heatmap = ax.imshow(aggregated_activity[trials_to_plot, :length_to_plot, i_neuron], cmap='viridis', aspect='auto',
                            interpolation='nearest')
        elif align=='end':
            heatmap = ax.imshow(aggregated_activity[trials_to_plot, -length_to_plot:, i_neuron], cmap='viridis', aspect='auto',
                            interpolation='nearest')
        else:
            assert align is None
            heatmap = ax.imshow(aggregated_activity[trials_to_plot, :, i_neuron], cmap='viridis', aspect='auto',
                            interpolation='nearest')
        ax.set_title(f'Neuron {i_neuron}')
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('Trials')
        if window_size is not None:  # aggregated activity in a window around timestamp
            assert align is None
            ax.set_xticks(np.arange(0, window_size * 2 + 1, 1))
            ax.set_xticklabels(np.arange(-window_size, window_size + 1, 1))
        else:
            assert window_size is None
            if align == 'start': # aggregated activity aligned to timestamp_start, x ticks start from 0
                ax.set_xticks(np.arange(0, length_to_plot, 20))
                ax.set_xticklabels(np.arange(0, length_to_plot, 20))
            elif align == 'end': # aggregated activity aligned to timestamp_end, x ticks ends with 0
                ax.set_xticks(np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20)+length_to_plot)
                ax.set_xticklabels(
                    np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20))
            else:
                raise ValueError('align must be start or end')
        ax.set_yticks(np.arange(0, n_trials, 20))
        ax.set_yticklabels(np.arange(1, n_trials + 1, 20))

        # Add colorbar
        fig.colorbar(heatmap, ax=ax, orientation='vertical')
        
        if align=='start':
            # Plot average tuning curve
            avg_activity = np.nanmean(aggregated_activity[trials_to_plot, :length_to_plot, i_neuron], axis=0)
            # Plot standard error of the mean
            std_activity = np.nanstd(aggregated_activity[trials_to_plot, :length_to_plot, i_neuron], axis=0)
        elif align=='end':
            # Plot average tuning curve
            avg_activity = np.nanmean(aggregated_activity[trials_to_plot, -length_to_plot:, i_neuron], axis=0)
            # Plot standard error of the mean
            std_activity = np.nanstd(aggregated_activity[trials_to_plot, -length_to_plot:, i_neuron], axis=0)
        else:
            assert align is None
            # Plot average tuning curve
            avg_activity = np.nanmean(aggregated_activity[trials_to_plot, :, i_neuron], axis=0)
            # Plot standard error of the mean
            std_activity = np.nanstd(aggregated_activity[trials_to_plot, :, i_neuron], axis=0)
        ax2.plot(np.arange(len(avg_activity)), avg_activity, label='Average Activity')
        ax2.fill_between(np.arange(len(avg_activity)), avg_activity - std_activity,
                         avg_activity + std_activity, alpha=0.2)
        ax2.set_xlabel('Time (bins)')
        ax2.set_ylabel('Average Activity')

        if window_size is not None:  # aggregated activity in a window around timestamp
            assert align is None
            ax2.set_xticks(np.arange(0, window_size * 2 + 1, 1))
            ax2.set_xticklabels(np.arange(-window_size, window_size + 1, 1))
        else:
            assert window_size is None
            if align == 'start': # aggregated activity aligned to timestamp_start, x ticks start from 0
                ax2.set_xticks(np.arange(0, length_to_plot, 20))
                ax2.set_xticklabels(np.arange(0, length_to_plot, 20))
            elif align == 'end': # aggregated activity aligned to timestamp_end, x ticks starts with ends with 0
                ax2.set_xticks(np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20)+length_to_plot)
                ax2.set_xticklabels(np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20))
            else:
                raise ValueError('align must be start or end')

        ax2.legend()

        # Save the figure
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'neuron_{i_neuron}.png'))
        plt.close()

# overlap the tuning curves for cells that fire for reward or to-reward run to see if activity rescale
def plot_overlap_tuning_curves_for_two_aggregated_activity(aggregated_activity1, aggregated_activity2, label1, label2, neuron_idx, save_dir, align='start', window_size=None):
    # note that agg_act1 and agg_act2 may come from different trials, hence length (i.e. number of trials) may be different
    for i_neuron in neuron_idx:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        avg_activity1 = np.nanmean(aggregated_activity1[:, :, i_neuron], axis=0)
        std_activity1 = np.nanstd(aggregated_activity1[:, :, i_neuron], axis=0)
        avg_activity2 = np.nanmean(aggregated_activity2[:, :, i_neuron], axis=0)
        std_activity2 = np.nanstd(aggregated_activity2[:, :, i_neuron], axis=0)

        if window_size is None:  # different durations for avg_activity1 and 2
            length_to_plot = max(len(avg_activity1), len(avg_activity2))
            if align=='start':  # align the two curves at the start

                ax.plot(np.arange(len(avg_activity1)), avg_activity1, label=label1, color='blue')
                ax.fill_between(np.arange(len(avg_activity1)), avg_activity1 - std_activity1,
                                avg_activity1 + std_activity1, alpha=0.2, color='blue')
                ax.plot(np.arange(len(avg_activity2)), avg_activity2, label=label2, color='orange')
                ax.fill_between(np.arange(len(avg_activity2)), avg_activity2 - std_activity2,
                                avg_activity2 + std_activity2, alpha=0.2, color='orange')
                # add ax ticks
                ax.set_xticks(np.arange(0, length_to_plot, 20))
                ax.set_xticklabels(np.arange(0, length_to_plot, 20))

            elif align=='end':  # align the two curves at the end

                ax.plot(np.arange(-len(avg_activity1), 0)+length_to_plot, avg_activity1, label=label1, color='blue')
                ax.fill_between(np.arange(-len(avg_activity1), 0)+length_to_plot, avg_activity1 - std_activity1,
                                avg_activity1 + std_activity1, alpha=0.2, color='blue')
                ax.plot(np.arange(-len(avg_activity2), 0)+length_to_plot, avg_activity2, label=label2, color='orange')
                ax.fill_between(np.arange(-len(avg_activity2), 0)+length_to_plot, avg_activity2 - std_activity2,
                                avg_activity2 + std_activity2, alpha=0.2, color='orange')
                # add ax ticks
                ax.set_xticks(np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20)+length_to_plot)
                ax.set_xticklabels(
                    np.arange(-length_to_plot - (-length_to_plot % -20), 1, 20))
        else:
            ax.plot(np.arange(window_size*2+1), avg_activity1, label=label1, color='blue')
            ax.fill_between(np.arange(window_size*2+1), avg_activity1 - std_activity1,
                            avg_activity1 + std_activity1, alpha=0.2, color='blue')
            ax.plot(np.arange(window_size*2+1), avg_activity2, label=label2, color='orange')
            ax.fill_between(np.arange(window_size*2+1), avg_activity2 - std_activity2,
                            avg_activity2 + std_activity2, alpha=0.2, color='orange')
            ax.set_xticks(np.arange(0, window_size * 2 + 1, 1))
            ax.set_xticklabels(np.arange(-window_size, window_size + 1, 1))
        ax.set_ylabel('Average Activity')
        ax.set_xlabel('Time (bins)')
        ax.legend()
        # Save the figure
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'neuron_{i_neuron}.png'))
        plt.close()


