import numpy as np
from scipy import stats
import itertools
from analysis.mutual_info.utils import shuffle_activity
import sklearn
import scikit_posthocs as sp


def lin_reg_ramping(resp):
    """
    Shikano et al., 2021; Toso et al., 2021
    Fit linear regression to trial-averaged tuning curve of each neuron to identify ramping cells.
    :param resp: n_total_episodes x len_delay x n_neurons
    :return: r, p, slope (beta) of each neuron
    """
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    # r_result = np.zeros(n_neurons)
    p_result = np.zeros(n_neurons)
    slope_result = np.zeros(n_neurons)
    intercept_result = np.zeros(n_neurons)
    R_result = np.zeros(n_neurons)
    for i_neuron in range(n_neurons):
        tuning_curve = np.mean(resp[:, :, i_neuron], axis=0)
        t = np.arange(len_delay)
        slope_result[i_neuron], intercept_result[i_neuron], r, p_result[i_neuron], std_err = stats.linregress(t,tuning_curve)
        R_result[i_neuron], pval = stats.pearsonr(t, tuning_curve)

    return p_result, slope_result, intercept_result, R_result


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


def ridge_to_background(resp, ramping_bool, percentile=95, n_shuff=1000):
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

    for i_neuron in range(n_neurons):
        trial_avg_resp = np.mean(resp[:, :, i_neuron], axis=0)  # (len_delay, )
        if ramping_bool[i_neuron]:  # if i_neuron is a ramping neuron
            trial_avg_resp = np.mean(resp[:, :, i_neuron], axis=0)
            t = np.arange(len_delay)
            slope, intercept, r, p, std_err = stats.linregress(t,trial_avg_resp)
            trial_avg_resp -= slope * t + intercept
        RB_result[i_neuron] = np.max(trial_avg_resp) / np.mean(trial_avg_resp)
        shuffled_RB = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            np.random.shuffle(trial_avg_resp)
            shuffled_RB[i_shuff] = np.max(trial_avg_resp) / np.mean(trial_avg_resp)
        z_RB_threshold_result[i_neuron] = np.percentile(shuffled_RB, percentile)

    return RB_result, z_RB_threshold_result


def trial_reliability_score(resp, split='odd-even', percentile=95, n_shuff=1000):
    """
    Yong et al., 2021
    :param resp: n_total_episodes x len_delay x n_neurons
    :param split: 'random' or 'odd-even'
    :param percentile: default=95
    :param n_shuff: default=1000
    :return:
    """
    assert split=='random' or split=='odd-even', "split must be random or odd-even"
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    trial_reliability_score_result = np.zeros(n_neurons)  # pearson corr coeff between the trial averaged response in the two trial splits
    trial_reliability_score_threshold_result = np.zeros(n_neurons)
    for i_neuron in range(n_neurons):
        if split == 'random':
            split_1_idx = np.random.choice(n_total_episodes, n_total_episodes//2, replace=False)
            ind = np.zeros(n_total_episodes, dtype=bool)
            ind[split_1_idx] = True
            rest = ~ind
            split_2_idx = np.arange(n_total_episodes)[rest]

        elif split == 'odd-even':
            split_1_idx = np.arange(start=0, stop=n_total_episodes, step=2)
            split_2_idx = np.arange(start=1, stop=n_total_episodes+1, step=2)

        resp_1 = resp[split_1_idx, :, i_neuron]  # (n_total_episodes/2, len_delay)
        resp_2 = resp[split_2_idx, :, i_neuron]
        trial_reliability_score_result[i_neuron], pval = stats.pearsonr(np.mean(resp_1, axis=0), np.mean(resp_2, axis=0))

        shuffled_score = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity(resp)
            resp_1 = shuffled_resp[split_1_idx, :, i_neuron]
            resp_2 = shuffled_resp[split_2_idx, :, i_neuron]
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
    for i_neuron in range(n_neurons):
        p_t = 1 / len_delay
        trial_avg_resp = np.mean(resp[:, :, i_neuron], axis=0)  # (len_delay, )
        I_result[i_neuron] = np.sum(trial_avg_resp * (p_t * np.log2(trial_avg_resp / np.mean(trial_avg_resp))))

        I_surrogate = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity(resp)
            trial_avg_resp = np.mean(shuffled_resp[:, :, i_neuron], axis=0)  # (len_delay, )
            I_surrogate[i_shuff] = np.sum(trial_avg_resp * (p_t * np.log2(trial_avg_resp / np.mean(trial_avg_resp))))

        I_threshold_result[i_neuron] = np.percentile(I_surrogate, percentile)

    return I_result, I_threshold_result


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










