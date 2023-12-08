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
from memory_profiler import profile

#===================================================================================================
def circular_shuffle_neural_activity(neuron_activity):
    T_episode = neuron_activity.size
    shift = np.random.randint(T_episode)
    return np.roll(neuron_activity, shift)

def swap_shuffle_neural_activity(trial_neural_activity):
    # trial_neural_activity: n_neurons x T_episode
    n_neurons, T_episode = trial_neural_activity.shape

    # Generate a random permutation for each neuron and apply it
    permutations = np.apply_along_axis(np.random.permutation, 1, np.arange(T_episode))
    shuffled_activity = trial_neural_activity[np.arange(n_neurons)[:, None], permutations]

    return shuffled_activity

#===================================================================================================

# identify if neuron's average activity at timestamp across trials is significantly (>99 percentile) higher than circularly shuffled average activity of this neuron at this timestamp across trials
def get_trial_average_value_at_time_step(neural_activity, trial_idx, timestamp, n_neurons=256):
    # Note: this function does NOT need neural_activity to be padded
    n_trials = len(trial_idx)
    assert n_trials == len(timestamp)
    trial_average_value = np.zeros(n_neurons)
    # neural_activity: n_episodes, each element is a list of length T_episode, each element is a numpy array of shape (n_neurons,)
    for i_neuron in tqdm(range(n_neurons)):
        trial_average_value[i_neuron] = np.mean([neural_activity[trial_idx[i_trial]][timestamp[i_trial]][i_neuron] for i_trial in range(n_trials)])
    return trial_average_value  # n_neurons


def get_trial_average_value_within_window(neural_activity, trial_idx, timestamp, window_size=5):
    # Ensure neural_activity is a NumPy array padded with NaNs
    windows = np.stack([
        np.nanmean(neural_activity[idx, :, max(0, ts - window_size): min(neural_activity.shape[2], ts + window_size + 1)], axis=1)
        for idx, ts in zip(trial_idx, timestamp)
    ], axis=0)
    trial_average_value = np.nanmean(windows, axis=0)  # (n_neurons,)
    return trial_average_value

def get_trial_average_value_between_timestamps(neural_activity, trial_idx, timestamp_start, timestamp_end):
    # This function assumes that 'neural_activity' is padded with NaNs to handle variable lengths
    n_neurons = neural_activity.shape[1]
    trial_averages = np.zeros((len(trial_idx), n_neurons))

    for i, (start, end) in enumerate(zip(timestamp_start, timestamp_end)):
        trial_data = neural_activity[trial_idx[i], :, start:end]
        trial_averages[i] = np.nanmean(trial_data, axis=1)

    return np.nanmean(trial_averages, axis=0)


def get_trial_average_value_at_time_step_shuffled(neural_activity, trial_idx, timestamp, shuffle='circular'):
    assert len(trial_idx) == len(timestamp)
    shuffled_activity = neural_activity[trial_idx, :, :]
    # Note: neural_activity must be padded with NaNs
    if shuffle == 'circular':
        shuffled_activity = np.apply_along_axis(circular_shuffle_neural_activity, 2, shuffled_activity)
    elif shuffle == 'swap':
        # Apply swap shuffle across all selected trials  # TODO: check if it actually works
        shuffled_activity = np.apply_along_axis(swap_shuffle_neural_activity, 2, shuffled_activity)
    # Extract values at specific timestamps and compute mean.
    shuffled_activity_at_timestamp = np.asarray([shuffled_activity[i, :, ts] for i, ts in enumerate(timestamp)])  # n_trials, n_neurons
    return np.nanmean(shuffled_activity_at_timestamp, axis=0)  # (n_neurons,)


def get_trial_average_value_within_window_shuffled(neural_activity, trial_idx, timestamp, window_size=5, shuffle='circular'):
    # This function assumes that 'neural_activity' is padded with NaNs to handle variable lengths
    # Shuffle neural_activity before extracting windows
    if shuffle == 'circular':
        shuffled_activity = np.apply_along_axis(circular_shuffle_neural_activity, 2, neural_activity[trial_idx, :, :])
    elif shuffle == 'swap':
        shuffled_activity = np.apply_along_axis(swap_shuffle_neural_activity, 2, neural_activity[trial_idx, :, :])
    windows = np.stack([
        np.nanmean(shuffled_activity[i, :, max(0, ts - window_size): min(shuffled_activity.shape[2], ts + window_size + 1)], axis=1)
        for i, ts in enumerate(timestamp)
    ], axis=0)
    trial_average_value_shuffled = np.nanmean(windows, axis=0)  # (n_neurons,)
    return trial_average_value_shuffled


def get_trial_average_value_between_timestamps_shuffled(neural_activity, trial_idx, timestamp_start, timestamp_end, shuffle='circular'):
    assert len(trial_idx) == len(timestamp_start) == len(timestamp_end)
    # This function assumes that 'neural_activity' is padded with NaNs to handle variable lengths
    # step 1: shuffle neural_activity
    shuffled_activity = neural_activity[trial_idx, :, :]
    if shuffle == 'circular':
        shuffled_activity = np.apply_along_axis(circular_shuffle_neural_activity, 2, shuffled_activity)
    elif shuffle == 'swap':
        # Apply swap shuffle across all selected trials  # TODO: check if it actually works
        shuffled_activity = np.apply_along_axis(swap_shuffle_neural_activity, 2, shuffled_activity)

    n_neurons = shuffled_activity.shape[1]
    trial_averages = np.zeros((len(trial_idx), n_neurons))

    for i, (start, end) in enumerate(zip(timestamp_start, timestamp_end)):
        trial_data = neural_activity[i, :, start:end]
        trial_averages[i] = np.nanmean(trial_data, axis=1)

    return np.nanmean(trial_averages, axis=0)


def identify_significant_neurons(neural_activity, neural_activity_padded, trial_idx, timestamp, n_shuffle=1000, percentile=99,
                                 shuffle='circular', plot=True, save_dir=None, use_window=True):
    if use_window:
        trial_average_value = get_trial_average_value_within_window(neural_activity, trial_idx, timestamp, window_size=5)
        shuffled_values = np.array(
            [get_trial_average_value_within_window_shuffled(neural_activity_padded, trial_idx, timestamp, shuffle=shuffle, window_size=5) for _ in
             tqdm(range(n_shuffle))])
    else:
        trial_average_value = get_trial_average_value_at_time_step(neural_activity, trial_idx, timestamp)
        shuffled_values = np.array(
            [get_trial_average_value_at_time_step_shuffled(neural_activity_padded, trial_idx, timestamp, shuffle) for _ in
             tqdm(range(n_shuffle))])  # n_shuffle x n_neurons
    significant_threshold = np.percentile(shuffled_values, percentile, axis=0)
    n_neurons = len(trial_average_value)
    if plot:
        # for each neuron, plot trial_average_value_shuffled as a histogram, and plot trial_average_value as a vertical line, and plot percentile as a vertical line
        for i_neuron in range(n_neurons):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(shuffled_values[:, i_neuron], bins=20)
            ax.axvline(trial_average_value[i_neuron], color='red')
            ax.axvline(significant_threshold[i_neuron], color='black')
            ax.set_xlabel('Average Activity')
            ax.set_ylabel('Count')
            ax.set_title(f'Neuron {i_neuron}')
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'neuron_{i_neuron}.png'))
                plt.close()
    return np.where(trial_average_value > significant_threshold)[0]


def identify_significant_neurons_between_timestamps(neural_activity, trial_idx, timestamp_start, timestamp_end,
                                                    n_shuffle=1000, percentile=99, shuffle='circular', plot=True, save_dir=False):
    # Get the average values for the actual data
    trial_average_value = get_trial_average_value_between_timestamps(neural_activity, trial_idx, timestamp_start,
                                                                     timestamp_end)

    # Prepare an array to hold the shuffled averages
    trial_average_value_shuffled = np.zeros((n_shuffle, len(trial_average_value)))

    # Shuffle and compute average values in a vectorized manner
    for i_shuffle in tqdm(range(n_shuffle)):
        shuffled_values = get_trial_average_value_between_timestamps_shuffled(neural_activity, trial_idx,
                                                                              timestamp_start, timestamp_end,
                                                                              shuffle=shuffle)
        trial_average_value_shuffled[i_shuffle] = shuffled_values

    # Compute the significant threshold for each neuron
    significant_threshold = np.percentile(trial_average_value_shuffled, percentile, axis=0)
    n_neurons = len(trial_average_value)

    if plot:
        # for each neuron, plot trial_average_value_shuffled as a histogram, and plot trial_average_value as a vertical line, and plot percentile as a vertical line
        for i_neuron in range(n_neurons):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(trial_average_value_shuffled[:, i_neuron], bins=20)
            ax.axvline(trial_average_value[i_neuron], color='red')
            ax.axvline(significant_threshold[i_neuron], color='black')
            ax.set_xlabel('Average Activity')
            ax.set_ylabel('Count')
            ax.set_title(f'Neuron {i_neuron}')
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'neuron_{i_neuron}.png'))
                plt.close()
    # Identify and return indices of significant neurons
    return np.where(trial_average_value > significant_threshold)[0]


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


