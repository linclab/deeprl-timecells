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


def calculate_tuning_curves(resp):
    """
    calculate tuning curves (trial-averaged response) for each neuron, assuming len_delay is the same for all trials.
    :param resp: n_total_episodes x len_delay x n_neurons
    :return: n_neurons x len_delay
    """
    return np.mean(resp, axis=0).T


def calculate_tuning_curves_varying_duration(resp):
    """
    calculate tuning curves (trial-averaged response) for each neuron, with trials with varying len_delay.
    :param resp: n_total_episodes x len_delay x n_neurons. Contains 0 for empty elements.
    :return: n_neurons x len_delay
    """
    new_resp = resp
    new_resp[resp==0] = np.nan
    return np.nanmean(new_resp, axis=0).T


def calculate_tuning_curves_single_neuron(resp):
    """
    calculate tuning curves (trial-averaged response) for each neuron, assuming len_delay is the same for all trials.
    :param resp: n_total_episodes x len_delay
    :return: len_delay
    """
    return np.mean(resp, axis=0)


def calculate_tuning_curves_single_neuron_varying_duration(resp, return_trimmed=True):
    """
    calculate tuning curves (trial-averaged response) for each neuron, with trials with varying len_delay.
    :param resp: n_total_episodes x len_delay. Contains 0 for empty elements.
    :return: len_delay (may contain NaN).  if return_trimmed, trim off NaN.
    """
    new_resp = resp
    new_resp[resp==0] = np.nan
    tuning_curve = np.nanmean(new_resp, axis=0)
    if return_trimmed:
        return tuning_curve[~np.isnan(tuning_curve)]
    else:
        return tuning_curve


def correlation_between_tuning_curves(tuning_curve_A, tuning_curve_B):
    """
    calculate the pearson r between the two tuning curves.
    :param tuning_curve_A: len_delay_A
    :param tuning_curve_B: len_delay_B
    :return: r (float), pval (float)
    """
    if len(tuning_curve_A) == len(tuning_curve_B):
        r, pval = stats.pearsonr(tuning_curve_A, tuning_curve_B)
    elif len(tuning_curve_A) < len(tuning_curve_B):
        r, pval = stats.pearsonr(tuning_curve_A, tuning_curve_B[:len(tuning_curve_A)])
    else:
        r, pval = stats.pearsonr(tuning_curve_A[:len(tuning_curve_B)], tuning_curve_A)
    return r, pval


def lin_reg_ramping(resp, plot=False, save_dir=None, title=None):
    """
    Shikano et al., 2021; Toso et al., 2021
    Fit linear regression to trial-averaged tuning curve of each neuron to identify ramping cells.
    :param resp: n_total_episodes x len_delay x n_neurons
    :return: r, p, slope (beta) of each neuron
    """
    n_total_episodes = np.shape(resp)[0]
    random_trial_id = np.random.choice(n_total_episodes, 10)
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    # r_result = np.zeros(n_neurons)
    p_result = np.zeros(n_neurons)
    slope_result = np.zeros(n_neurons)
    intercept_result = np.zeros(n_neurons)
    pearson_R_result = np.zeros(n_neurons)
    tuning_curves = calculate_tuning_curves(resp)
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('tuning_curves', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = tuning_curves[i_neuron]
        t = np.arange(len_delay)
        slope_result[i_neuron], intercept_result[i_neuron], pearson_R_result[i_neuron], p_result[i_neuron], std_err = stats.linregress(t,tuning_curve)

        if plot:
            random_trial_resp_to_plot = resp[random_trial_id, :, i_neuron]
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            tuning_curve_plot.plot(tuning_curve, color='blue', alpha=1)  # TODO: remove boundary when running on cluster
            for i_trial in range(len(random_trial_resp_to_plot)):
                tuning_curve_plot.plot(random_trial_resp_to_plot[i_trial], alpha=0.1,
                                       color='red' if np.logical_and(p_result[i_neuron]<=0.05, np.abs(pearson_R_result[i_neuron])>=0.9) else 'blue')
            tuning_curve_plot.set_title(f'p={p_result[i_neuron]:.5f}\nr={pearson_R_result[i_neuron]:.5f}')
            # tuning_curve_plot.legend()
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curves_for_ramping_identification.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curves_for_ramping_identification.pdf saved to {save_dir}')
            except:
                pass
    return p_result, slope_result, intercept_result, pearson_R_result


def ridge_to_background(resp, ramping_bool, percentile=95, n_shuff=1000, plot=False, save_dir=None, title=None):
    """
    Toso et al., 2021
    Note: resp should be normalized for each neuron, 0 = this neuron's mininum resp, 1 = this neuron's maximum resp
    :param resp: n_total_episodes x len_delay x n_neurons
    :param ramping_bool: (n_neurons, ) array of boolean, indicated whether each neuron is a ramping neuron
    :param percentile: percentile cutoff above which a neuron is categorized as having significant RB (Toso et al., 2021). Default=95
    :param n_shuff: Default=1000
    :return: RB_result (n_neurons,), z_RB_threshold_result (n_neurons, )
    """
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    RB_result = np.zeros(n_neurons)
    z_RB_threshold_result = np.zeros(n_neurons)
    tuning_curves = calculate_tuning_curves(resp)
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('ramp_subtracted', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = tuning_curves[i_neuron]  # (len_delay, )
        if ramping_bool[i_neuron]:  # if i_neuron is a ramping neuron, then subtract the linear regression
            t = np.arange(len_delay)
            slope, intercept, r, p, std_err = stats.linregress(t, tuning_curve)
            lin_reg = slope * t + intercept
            lin_subtracted_tuning_curve = tuning_curve - (slope * t + intercept)

        if ramping_bool[i_neuron]:
            RB_result[i_neuron] = np.max(lin_subtracted_tuning_curve) / np.mean(lin_subtracted_tuning_curve)
            shuffled_RB = np.zeros(n_shuff)
            lin_subtracted_resp = resp[:, :, i_neuron] - np.tile(lin_reg, (n_total_episodes, 1))
            for i_shuff in range(n_shuff):
                shuffled_resp = shuffle_activity_single_neuron(lin_subtracted_resp)
                new_tuning_curve = calculate_tuning_curves_single_neuron(shuffled_resp)
                shuffled_RB[i_shuff] = np.max(new_tuning_curve) / np.mean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)
        else:
            RB_result[i_neuron] = np.max(tuning_curve) / np.mean(tuning_curve)
            shuffled_RB = np.zeros(n_shuff)
            for i_shuff in range(n_shuff):
                shuffled_resp = shuffle_activity_single_neuron(resp[:, :, i_neuron])
                new_tuning_curve = calculate_tuning_curves_single_neuron(shuffled_resp)
                shuffled_RB[i_shuff] = np.max(new_tuning_curve) / np.mean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)

        if plot:
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            if ramping_bool[i_neuron]:
                tuning_curve_plot.plot(tuning_curve, color='grey', alpha=1)
                tuning_curve_plot.plot(lin_reg, color='grey', alpha=0.25)
                tuning_curve_plot.plot(lin_subtracted_tuning_curve,
                                       color='red' if RB_result[i_neuron] > z_RB_threshold_result[i_neuron] else 'blue', alpha=1)
            else:
                tuning_curve_plot.plot(tuning_curve,
                                       color='red' if RB_result[i_neuron] > z_RB_threshold_result[i_neuron] else 'blue', alpha=1)
            tuning_curve_plot.set_title(f'RB={RB_result[i_neuron]:.5f}\nz_RB={z_RB_threshold_result[i_neuron]:.5f}')
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curve_for_seq_identification.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curve_for_seq_identification.pdf saved to {save_dir}')
            except:
                pass

    return RB_result, z_RB_threshold_result


def trial_reliability_vs_shuffle_score(resp, split='odd-even', percentile=95, n_shuff=1000):
    """
    Yong et al., 2021
    :param resp: n_total_episodes x len_delay x n_neurons
    :param split: 'random' or 'odd-even'
    :param percentile: default=95
    :param n_shuff: default=1000
    :return: trial_reliability_score_result (n_neurons,) , trial_reliability_score_threshold_result (n_neurons,)
    """
    assert split=='random' or split=='odd-even', "split must be random or odd-even"
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    trial_reliability_score_result = np.zeros(n_neurons)  # pearson corr coeff between the trial averaged response in the two trial splits
    trial_reliability_score_threshold_result = np.zeros(n_neurons)
    for i_neuron in tqdm(range(n_neurons)):
        if split == 'random':
            split_1_idx = np.random.choice(n_total_episodes, n_total_episodes//2, replace=False)
            ind = np.zeros(n_total_episodes, dtype=bool)
            ind[split_1_idx] = True
            rest = ~ind
            split_2_idx = np.arange(n_total_episodes)[rest]

        elif split == 'odd-even':
            split_1_idx = np.arange(start=0, stop=n_total_episodes-1, step=2)
            split_2_idx = np.arange(start=1, stop=n_total_episodes, step=2)

        resp_1 = resp[split_1_idx, :, i_neuron]  # (n_total_episodes/2, len_delay)
        resp_2 = resp[split_2_idx, :, i_neuron]
        trial_reliability_score_result[i_neuron], pval = stats.pearsonr(np.mean(resp_1, axis=0), np.mean(resp_2, axis=0))

        shuffled_score = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity_single_neuron(resp[:, :, i_neuron]) # gives higher shuffled reliability score
            resp_1 = shuffled_resp[split_1_idx, :]
            resp_2 = shuffled_resp[split_2_idx, :]
            shuffled_score[i_shuff], pval = stats.pearsonr(np.mean(resp_1, axis=0), np.mean(resp_2, axis=0))

        trial_reliability_score_threshold_result[i_neuron] = np.percentile(shuffled_score, percentile)

    return trial_reliability_score_result, trial_reliability_score_threshold_result


# For identifying ramping cells and time cells in timing task

def lin_reg_ramping_varying_duration(resp, plot=False, save_dir=None, title=None):
    """
    Shikano et al., 2021; Toso et al., 2021
    Fit linear regression to trial-averaged tuning curve of each neuron to identify ramping cells.
    :param resp: n_total_episodes x len_delay x n_neurons
    :return: r, p, slope (beta) of each neuron
    """
    n_total_episodes = np.shape(resp)[0]
    random_trial_id = np.random.choice(n_total_episodes, 10)
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    # r_result = np.zeros(n_neurons)
    p_result = np.zeros(n_neurons)
    slope_result = np.zeros(n_neurons)
    intercept_result = np.zeros(n_neurons)
    pearson_R_result = np.zeros(n_neurons)
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('tuning_curves', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = calculate_tuning_curves_single_neuron_varying_duration(resp[:, :, i_neuron], return_trimmed=True)
        t = np.arange(len(tuning_curve))
        slope_result[i_neuron], intercept_result[i_neuron], pearson_R_result[i_neuron], p_result[i_neuron], std_err = stats.linregress(t,tuning_curve)

        if plot:
            random_trial_resp_to_plot = resp[random_trial_id, :, i_neuron]
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            tuning_curve_plot.plot(tuning_curve, color='blue', alpha=1)
            for i_trial in range(len(random_trial_resp_to_plot)):
                tuning_curve_plot.plot(random_trial_resp_to_plot[i_trial], alpha=0.1,
                                       color='red' if np.logical_and(p_result[i_neuron]<=0.05, np.abs(pearson_R_result[i_neuron])>=0.9) else 'blue')
            tuning_curve_plot.set_title(f'p={p_result[i_neuron]:.5f}\nr={pearson_R_result[i_neuron]:.5f}')
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curves_for_ramping_identification.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curves_for_ramping_identification.pdf saved to {save_dir}')
            except:
                pass
    return p_result, slope_result, intercept_result, pearson_R_result


def ridge_to_background_varying_duration(resp, stim_duration, ramping_bool, percentile=95, n_shuff=1000, plot=False, save_dir=None, title=None):
    """
    Toso et al., 2021
    Note: resp should be normalized for each neuron, 0 = this neuron's mininum resp, 1 = this neuron's maximum resp
    :param resp: n_total_episodes x len_delay x n_neurons
    :param ramping_bool: (n_neurons, ) array of boolean, indicated whether each neuron is a ramping neuron
    :param percentile: percentile cutoff above which a neuron is categorized as having significant RB (Toso et al., 2021). Default=95
    :param n_shuff: Default=1000
    :return: RB_result (n_neurons,), z_RB_threshold_result (n_neurons, )
    """
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    RB_result = np.zeros(n_neurons)
    z_RB_threshold_result = np.zeros(n_neurons)
    nan_resp = resp
    nan_resp[resp==0] = np.nan
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('ramp_subtracted', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = calculate_tuning_curves_single_neuron_varying_duration(resp, return_trimmed=True)
        if ramping_bool[i_neuron]:  # if i_neuron is a ramping neuron, then subtract the linear regression
            t = np.arange(len(tuning_curve))
            slope, intercept, r, p, std_err = stats.linregress(t, tuning_curve)
            lin_reg = slope * t + intercept
            lin_subtracted_tuning_curve = tuning_curve - (slope * t + intercept)

        if ramping_bool[i_neuron]:
            RB_result[i_neuron] = np.nanmax(lin_subtracted_tuning_curve) / np.nanmean(lin_subtracted_tuning_curve)
            shuffled_RB = np.zeros(n_shuff)
            lin_subtracted_resp = nan_resp[:, :len(tuning_curve), i_neuron] - np.tile(lin_reg, (n_total_episodes, 1))
            for i_shuff in range(n_shuff):
                shuffled_resp = shuffle_activity_single_neuron_varying_duration(lin_subtracted_resp, stim_duration, return_nan=False)  # contains 0
                new_tuning_curve = calculate_tuning_curves_single_neuron_varying_duration(shuffled_resp)
                shuffled_RB[i_shuff] = np.nanmax(new_tuning_curve) / np.nanmean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)
        else:
            RB_result[i_neuron] = np.nanmax(tuning_curve) / np.nanmean(tuning_curve)
            shuffled_RB = np.zeros(n_shuff)
            for i_shuff in range(n_shuff):
                shuffled_resp = shuffle_activity_single_neuron_varying_duration(resp[:,:,i_neuron], stim_duration, return_nan=False)
                new_tuning_curve = calculate_tuning_curves_single_neuron_varying_duration(shuffled_resp)
                shuffled_RB[i_shuff] = np.nanmax(new_tuning_curve) / np.nanmean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)

        if plot:
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            if ramping_bool[i_neuron]:
                tuning_curve_plot.plot(tuning_curve, color='grey', alpha=1)
                tuning_curve_plot.plot(lin_reg, color='grey', alpha=0.25)
                tuning_curve_plot.plot(lin_subtracted_tuning_curve,
                                       color='red' if RB_result[i_neuron] > z_RB_threshold_result[i_neuron] else 'blue', alpha=1)
            else:
                tuning_curve_plot.plot(tuning_curve,
                                       color='red' if RB_result[i_neuron] > z_RB_threshold_result[i_neuron] else 'blue', alpha=1)
            tuning_curve_plot.set_title(f'RB={RB_result[i_neuron]:.5f}\nz_RB={z_RB_threshold_result[i_neuron]:.5f}')
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curve_for_seq_identification.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curve_for_seq_identification.pdf saved to {save_dir}')
            except:
                pass

    return RB_result, z_RB_threshold_result


def trial_reliability_vs_shuffle_score_varying_duration(resp, stim, split='odd-even', percentile=95, n_shuff=1000):
    """
    Yong et al., 2021
    :param resp: n_total_episodes x len_delay x n_neurons
    :param split: 'random' or 'odd-even'
    :param percentile: default=95
    :param n_shuff: default=1000
    :return: trial_reliability_score_result (n_neurons,) , trial_reliability_score_threshold_result (n_neurons,)
    """
    assert split=='random' or split=='odd-even', "split must be random or odd-even"
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    trial_reliability_score_result = np.zeros(n_neurons)  # pearson corr coeff between the trial averaged response in the two trial splits
    trial_reliability_score_threshold_result = np.zeros(n_neurons)
    for i_neuron in tqdm(range(n_neurons)):
        if split == 'random':
            split_1_idx = np.random.choice(n_total_episodes, n_total_episodes//2, replace=False)
            ind = np.zeros(n_total_episodes, dtype=bool)
            ind[split_1_idx] = True
            rest = ~ind
            split_2_idx = np.arange(n_total_episodes)[rest]

        elif split == 'odd-even':
            split_1_idx = np.arange(start=0, stop=n_total_episodes-1, step=2)
            split_2_idx = np.arange(start=1, stop=n_total_episodes, step=2)

        resp_1 = resp[split_1_idx, :, i_neuron]  # (n_total_episodes/2, len_delay)
        resp_2 = resp[split_2_idx, :, i_neuron]
        trial_reliability_score_result[i_neuron], pval = correlation_between_tuning_curves(
            calculate_tuning_curves_single_neuron_varying_duration(resp_1),
            calculate_tuning_curves_single_neuron_varying_duration(resp_2))

        shuffled_score = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity_single_neuron_varying_duration(resp[:, :, i_neuron], stim_duration=stim, return_nan=False) # gives higher shuffled reliability score
            resp_1 = shuffled_resp[split_1_idx, :]  # TODO: try shuffling splitting too
            resp_2 = shuffled_resp[split_2_idx, :]
            shuffled_score[i_shuff], pval = correlation_between_tuning_curves(
                calculate_tuning_curves_single_neuron_varying_duration(resp_1),
                calculate_tuning_curves_single_neuron_varying_duration(resp_2))

        trial_reliability_score_threshold_result[i_neuron] = np.percentile(shuffled_score, percentile)

    return trial_reliability_score_result, trial_reliability_score_threshold_result


def sort_response_by_peak_latency(total_resp, cell_nums_ramp, cell_nums_seq, norm=True):
    """
    Average the responses across episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and sort cells by when their maximum response happens.
    Then, Separate cells into ramping cells (strictly increasing/decreasing) and sequence cells.
    Note: sequence cells may contain NaN rows.
    - Arguments: total_resp, norm=True
    - Returns: cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))  # len(segments) is also len_delay
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    for i in range(len(segments)):  # at timestep i
        averages = np.mean(segments[i],
                           axis=0)  # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
        unsorted_matrix[:, i] = np.transpose(
            averages)  # goes into the i-th column of unsorted_matrix, each row is one neuron
        if norm is True:
            normalized_matrix = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) / np.ptp(
                unsorted_matrix, axis=1, keepdims=True)
            # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
            max_indeces = np.argmax(normalized_matrix, axis=1)  # which time step does the maximum firing occur
            cell_nums = np.argsort(max_indeces)  # returns the order of cell number that should go into sorted_matrix
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = normalized_matrix[i_cell]
        else:
            max_indeces = np.argmax(unsorted_matrix, axis=1)  # which time step does the maximum firing occur
            cell_nums = np.argsort(max_indeces)  # returns the order of cell number that should go into sorted_matrix
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = unsorted_matrix[i_cell]
    # At this point, sorted_matrix should contain all cells
    assert len(sorted_matrix) == n_neurons

    sorted_matrix_seq = sorted_matrix[np.isin(cell_nums, cell_nums_seq)]
    sorted_matrix_ramp = sorted_matrix[np.isin(cell_nums, cell_nums_ramp)]
    sorted_matrix_nontime = sorted_matrix[~np.isin(cell_nums, np.intersect1d(cell_nums_ramp, cell_nums_ramp))]

    # Re-arrange cell_nums_seq and cell_num_ramp and cell_nums_nontime according to peak latency
    cell_nums_seq = cell_nums[np.isin(cell_nums, cell_nums_seq)]
    cell_nums_ramp = cell_nums[np.isin(cell_nums, cell_nums_ramp)]
    cell_nums_nontime = cell_nums[~np.isin(cell_nums, np.intersect1d(cell_nums_ramp, cell_nums_ramp))]

    return cell_nums, cell_nums_seq, cell_nums_ramp, cell_nums_nontime,\
           sorted_matrix, sorted_matrix_seq, sorted_matrix_ramp, sorted_matrix_nontime


def calculate_field_width(tuning_curve):
    """
    calculate the field width (width of half height of peak activity), which must include the time bin for peak activity
    :param tuning_curve: len_delay
    :return: field_width (int)
    """
    peak_time = np.argmax(tuning_curve)
    active_bool = tuning_curve > (peak_time/2)
    arr = np.where(np.concatenate(([active_bool[0]],active_bool[:-1] != active_bool[1:],[True])))[0]
    left_bound_idx = np.searchsorted(arr, peak_time) - 1
    right_bound_idx = np.searchsorted(arr, peak_time)
    field_width = arr[right_bound_idx] - arr[left_bound_idx]
    return peak_time, field_width


def plot_r_tuning_curves(resp_1, resp_2, label_1, label_2, save_dir, varying_duration=False):
    """
    plot the histogram of r(tuning_curves_1, tuning_curves_2).
    Need to have the same number of neurons.
    :param resp_1: n_episodes_1, len_delay, n_neurons
    :param resp_2: n_episodes_2, len_delay, n_neurons
    :param varying_delay: If resp contains trials with varying length. If true,
        tuning curves are calculated with a different function.
    :return: None
    """
    assert resp_1.shape[-1] == resp_2.shape[-1], "resp_1 and resp_2 must have the same number of neurons"
    if varying_duration:
        r = correlation_between_tuning_curves(
            calculate_tuning_curves_varying_duration(resp_1),
            calculate_tuning_curves_varying_duration(resp_2)
        )
    else:
        r = correlation_between_tuning_curves(
            calculate_tuning_curves(resp_1),
            calculate_tuning_curves(resp_2)
        )
    plt.figure()
    plt.hist(r, range=(-1,1), bins=50)
    plt.xlabel(f"r({label_1}, {label_2})")
    plt.ylabel("Fraction")
    plt.savefig(os.path.join(save_dir, f"{label_1}_{label_2}_r_hist.svg"))


def plot_field_width_vs_peak_time(resp, save_dir, title):
    """
    scatter plot field width vs peak time for each neuron. Trials must have same duration.
    :param resp: n_episodes x len_delay x n_neurons
    :return: None
    """
    tuning_curves = calculate_tuning_curves(resp)
    n_neurons = resp.shape[-1]
    field_widths = np.zeros(n_neurons)
    peak_times = np.zeros(n_neurons)
    for i_neuron in range(n_neurons):
        peak_times[i_neuron], field_widths[i_neuron] = calculate_field_width(tuning_curves[i_neuron])
    plt.figure()
    plt.scatter(peak_times, field_widths)
    slope, intercept, r, p, std_err = stats.linregress(x=peak_times, y=field_widths)
    lin_reg = slope * np.arange(20) + intercept
    plt.plot(lin_reg, alpha=0.5)
    plt.xlabel("Peak time")
    plt.ylabel("Field width")
    plt.savefig(os.path.join(save_dir, f"{title}p_eak_time_vs_field_width.svg"))


def identify_stimulus_selective_neurons(resp, stim_labels, alpha=0.01, varying_duration=False):
    """
    Identify stimulus-selective neurons based on their response to different stimuli using the method described
    in Toso et al. (2021). Returns a boolean array indicating which neurons are selective.

    To use this function, you would pass in an array resp of shape (n_neurons, n_trials) containing the neural response
    to different stimuli, and an array stim_labels of length n_trials containing labels for each stimulus. The function
    returns a boolean array is_selective indicating which neurons are selective based on a significance level alpha
    (default 0.01) for the ANOVA test. Stimulus-selective neurons are those for which the null hypothesis of equal mean
    responses across all stimuli is rejected.

    Parameters:
        resp: (n_total_trials x len_delay x n_neurons)
        stim_labels: (n_total_trials,)
        alpha (float, optional): Significance level for statistical test. Default is 0.01.

    Returns:
    is_selective (ndarray): Boolean array of length n_neurons indicating which neurons are selective.
    """

    # Get unique stimulus labels
    unique_labels = np.unique(stim_labels)  # 0, 1

    # Compute mean response to each stimulus for each neuron  # <-- tuning curves for L vs R
    mean_resp = np.zeros((resp.shape[0], resp.shape[1], len(unique_labels)))  # n_neurons x len_delay x 2
    for i, label in enumerate(unique_labels):
        if varying_duration:
            mean_resp[:, :, i] = calculate_tuning_curves_varying_duration(resp[stim_labels==label])
        else:
            mean_resp[:, :, i] = calculate_tuning_curves(resp[stim_labels==label])

    # Compute ANOVA for each neuron
    p_vals = np.zeros((resp.shape[0],))  # n_neurons
    for i in range(resp.shape[0]):  # for each neuron in n_neurons
        _, p_vals[i] = stats.f_oneway(*[resp[i, :, stim_labels == label] for label in unique_labels])

    # Identify selective neurons
    is_selective = (p_vals < alpha)

    return is_selective

#=================== TO CHUCK? =============================
def skaggs_temporal_information(resp, n_shuff=1000, percentile=95):
    """
    Heys & Dombeck, 2018
    :return: I_result, I_threshold_result
    """
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    I_result = np.zeros(n_neurons)
    I_threshold_result = np.zeros(n_neurons)
    for i_neuron in tqdm(range(n_neurons)):
        p_t = 1 / len_delay
        tuning_curve = np.mean(resp[:, :, i_neuron], axis=0)  # (len_delay, )
        I_result[i_neuron] = np.sum(tuning_curve * (p_t * np.log2(tuning_curve / np.mean(tuning_curve))))

        I_surrogate = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity_single_neuron(resp[:, :, i_neuron])
            tuning_curve = calculate_tuning_curves_single_neuron(shuffled_resp)
            I_surrogate[i_shuff] = np.sum(tuning_curve * (p_t * np.log2(tuning_curve / np.mean(tuning_curve))))

        I_threshold_result[i_neuron] = np.percentile(I_surrogate, percentile)

    return I_result, I_threshold_result


def trial_consistency_across_durations(stim, stim1_resp, stim2_resp, type='absolute'):
    stim_set = np.sort(np.unique(stim))
    n_neurons = np.shape(stim1_resp)[-1]
    stim_trial_avg_resp = []
    stim_combinations_list = list(itertools.combinations(stim_set, r=2))
    for stim_len in stim_set:
        stim1_activities = stim1_resp[stim[:, 0] == stim_len, :stim_len, :]
        stim2_activities = stim2_resp[stim[:, 0] == stim_len, :stim_len, :]
        stim_trial_avg_resp.append(np.mean(np.stack(stim1_activities, stim2_activities, axis=1), axis=0))  # (T1, n_neurons)

    R_result = np.zeros((n_neurons, len(stim_combinations_list)))
    pval_result = np.zeros((n_neurons, len(stim_combinations_list)))

    for i_neuron in tqdm(range(n_neurons)):
        for i_stim_comb, stim_combination in enumerate(stim_combinations_list):
            t1 = stim_combination[0]
            i_t1 = list(stim_set).index(t1)
            t2 = stim_combination[1]
            i_t2 = list(stim_set).index(t2)
            r_t1 = stim_trial_avg_resp[i_t1][:, i_neuron]
            r_t2 = stim_trial_avg_resp[i_t2][:, i_neuron]
            if type=='absolute':
                R_result[i_neuron, i_stim_comb], pval_result[i_neuron, i_stim_comb] = stats.pearsonr(r_t1, r_t2[:t1])
            elif type=='relative':
                R_result[i_neuron, i_stim_comb], pval_result[i_neuron, i_stim_comb] = stats.pearsonr(np.repeat(r_t1, t2), np.repeat(r_t2, t1))

    return R_result, pval_result


def skaggs_temporal_information_varying_duration(resp, stim, n_shuff=1000, percentile=95):
    """
    Heys & Dombeck, 2018
    :param resp: n_total_episodes x len_delay x n_neurons
    :param stim: stimulus array for stim1, stim2, OR delay. Shape: (n_total_episodes, )
    :param n_shuff: Default = 1000
    :param percentile: Default = 95
    :return:
    """
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    I_result = np.zeros(n_neurons)
    I_threshold_result = np.zeros(n_neurons)
    stim_set = np.sort(np.unique(stim))  # [10, 15, ..., 40]
    num_stim = np.max(np.shape(stim_set))  # 7

    for i_neuron in range(n_neurons):
        recep_field = np.zeros((n_total_episodes, int(np.max(stim_set))))  # (n_total_episodes, 40)
        recep_field[:, :] = np.nan
        start_idx = 0
        for stim_idx, stim_len in enumerate(range(num_stim)):
            episode_idx = np.where(stim == stim_len)[0]
            recep_field[start_idx:len(episode_idx), :stim_len] = resp[episode_idx, :stim_len, i_neuron]
            start_idx += len(episode_idx)

        I_result[i_neuron] = 0
        for t in range(len_delay):
            num_occurrence = np.count_nonzero(~np.isnan(recep_field[:, t]))
            num_total_occurrence = np.count_nonzero(~np.isnan(recep_field))
            p_t = num_occurrence / num_total_occurrence
            avg_resp_t = np.nanmean(recep_field[:, t])
            avg_resp = np.nanmean(recep_field)
            I_result[i_neuron] += p_t * avg_resp_t * np.log2(avg_resp_t / avg_resp)

        # shuffle recep_field and re-calculate I
        I_shuffle = np.zeros(n_shuff)
        for i_shuff in tqdm(range(n_shuff)):
            # Create shuffled recep_field
            shuff_recep_field = np.zeros((n_total_episodes, int(np.max(stim_set))))  # (n_total_episodes, 40)
            shuff_recep_field[:, :] = np.nan
            start_idx = 0
            for stim_idx, stim_len in enumerate(range(num_stim)):
                episode_idx = np.where(stim == stim_len)[0]
                shuff_recep_field[start_idx:len(episode_idx), :stim_len] = shuffle_activity(resp[episode_idx, :stim_len, i_neuron])
                start_idx += len(episode_idx)

            I_shuffle[i_shuff] = 0
            for t in range(len_delay):
                num_occurrence = np.count_nonzero(~np.isnan(shuff_recep_field[:, t]))
                num_total_occurrence = np.count_nonzero(~np.isnan(shuff_recep_field))
                p_t = num_occurrence / num_total_occurrence
                avg_resp_t = np.nanmean(shuff_recep_field[:, t])
                avg_resp = np.nanmean(shuff_recep_field)
                I_shuffle[i_shuff] += p_t * avg_resp_t * np.log2(avg_resp_t / avg_resp)

        I_threshold_result[i_neuron] = np.percentile(I_shuffle, percentile)

    return I_result, I_threshold_result


def identify_place_cells(resp, loc, n_shuff, percentile):
    from utils_mutual_info import construct_ratemap, construct_ratemap_occupancy
    n_neurons = resp.shape[-1]
    RB_arr = np.zeros(n_neurons)
    zRB_threshold_arr = np.zeros(n_neurons)
    is_place_cell = np.zeros(n_neurons)
    for i_neuron in range(n_neurons):
        # generate occupancy-normalized heatmap
        ratemap, spatial_occupancy = construct_ratemap(delay_resp=resp, delay_loc=loc, norm=True)
        # TODO: check if is ratemap occupancy-normalized here?
        ratemap /= spatial_occupancy
        RB = np.max(ratemap) / np.mean(ratemap)

        zRB = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_ratemap, shuffled_spatial_occupancy = construct_ratemap_occupancy(delay_resp=resp, delay_loc=loc, shuffle=True)
            shuffled_ratemap /= shuffled_spatial_occupancy
            zRB[i_shuff] = np.max(shuffled_ratemap) / np.mean(shuffled_ratemap)

        zRB_threshold_arr[i_neuron] = np.percentile(zRB, percentile)
        is_place_cell[i_neuron] = RB > zRB_threshold_arr
    return RB_arr, zRB_threshold_arr, is_place_cell