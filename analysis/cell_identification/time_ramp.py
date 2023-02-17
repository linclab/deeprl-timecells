import numpy as np
from scipy import stats
import itertools
from analysis.mutual_info.utils import shuffle_activity
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import sklearn
import scikit_posthocs as sp


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
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('tuning_curves', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = np.mean(resp[:, :, i_neuron], axis=0)
        t = np.arange(len_delay)
        slope_result[i_neuron], intercept_result[i_neuron], pearson_R_result[i_neuron], p_result[i_neuron], std_err = stats.linregress(t,tuning_curve)

        if plot:
            random_trial_resp_to_plot = resp[random_trial_id, :, i_neuron]
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            tuning_curve_plot.plot(tuning_curve, color='blue', alpha=1)
            for i_trial in range(len(random_trial_resp_to_plot)):
                tuning_curve_plot.plot(random_trial_resp_to_plot[i_trial], alpha=0.1, color='blue')
            tuning_curve_plot.set_title(f'p={p_result[i_neuron]:.5f}\nr={pearson_R_result[i_neuron]:.5f}')
            # tuning_curve_plot.legend()
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curves_for_ramping_identification.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curves.pdf saved to {save_dir}')
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
    if plot:
        plot_cols = 10
        plot_rows = int(np.ceil(n_neurons / plot_cols))
        tuning_curve_fig = plt.figure('ramp_subtracted', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)

    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = np.mean(resp[:, :, i_neuron], axis=0)  # (len_delay, )
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
                shuffled_resp = np.random.permutation(lin_subtracted_resp.flatten()).reshape((n_total_episodes, len_delay))  #TODO: HUGE
                new_tuning_curve = np.mean(shuffled_resp, axis=0)
                shuffled_RB[i_shuff] = np.max(new_tuning_curve) / np.mean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)
        else:
            RB_result[i_neuron] = np.max(tuning_curve) / np.mean(tuning_curve)
            shuffled_RB = np.zeros(n_shuff)
            for i_shuff in range(n_shuff):
                # shuffled_resp = np.random.permutation(resp[:, :, i_neuron].flatten()).reshape((n_total_episodes, len_delay))
                # new_tuning_curve = np.mean(shuffled_resp, axis=0)
                shuffled_resp = shuffle_activity(resp)
                new_tuning_curve = np.mean(shuffled_resp[:, :, i_neuron], axis=0)
                shuffled_RB[i_shuff] = np.max(new_tuning_curve) / np.mean(new_tuning_curve)
            z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)

        if plot:
            tuning_curve_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
            if ramping_bool[i_neuron]:
                tuning_curve_plot.plot(tuning_curve, color='grey', alpha=1)
                tuning_curve_plot.plot(lin_reg, color='grey', alpha=0.25)
                tuning_curve_plot.plot(lin_subtracted_tuning_curve, color='blue', alpha=1)
            else:
                tuning_curve_plot.plot(tuning_curve, color='blue', alpha=1)
            tuning_curve_plot.set_title(f'RB={RB_result[i_neuron]:.5f}\nz_RB={z_RB_threshold_result[i_neuron]:.5f}')
    if plot:
        with PdfPages(os.path.join(save_dir, f'{title}_tuning_curve_for_seq_identification_new.pdf')) as pdf:
            try:
                pdf.savefig(tuning_curve_fig)
                plt.close(tuning_curve_fig)
                print(f'{title}_tuning_curve_for_seq_identification.pdf saved to {save_dir}')
            except:
                pass

    return RB_result, z_RB_threshold_result


def trial_reliability_score(resp, split='odd-even', percentile=95, n_shuff=1000):
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
            shuffled_resp = shuffle_activity(resp) # gives higher shuffled reliability score
            resp_1 = shuffled_resp[split_1_idx, :, i_neuron]
            resp_2 = shuffled_resp[split_2_idx, :, i_neuron]
            # shuffled_resp = np.random.permutation(resp[:, :, i_neuron].flatten()).reshape((n_total_episodes, len_delay))
            # resp_1 = shuffled_resp[split_1_idx, :]
            # resp_2 = shuffled_resp[split_2_idx, :]
            shuffled_score[i_shuff], pval = stats.pearsonr(np.mean(resp_1, axis=0), np.mean(resp_2, axis=0))

        trial_reliability_score_threshold_result[i_neuron] = np.percentile(shuffled_score, percentile)

    return trial_reliability_score_result, trial_reliability_score_threshold_result


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
            shuffled_resp = shuffle_activity(resp)
            tuning_curve = np.mean(shuffled_resp[:, :, i_neuron], axis=0)  # (len_delay, )
            I_surrogate[i_shuff] = np.sum(tuning_curve * (p_t * np.log2(tuning_curve / np.mean(tuning_curve))))

        I_threshold_result[i_neuron] = np.percentile(I_surrogate, percentile)

    return I_result, I_threshold_result


# For identifying ramping cells and time cells in timing task
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

    for i_neuron in range(n_neurons):
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
        for i_shuff in range(n_shuff):
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




