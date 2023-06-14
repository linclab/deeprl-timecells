import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.manifold import TSNE
import torch
import numpy as np
import utils_linclab_plot
import os
from matplotlib_venn import venn2
import umap
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

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


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


def make_piechart(n_ramp_neurons, n_seq_neurons, n_neurons, save_dir, label, save=False):
    neuron_counts = np.array([n_ramp_neurons, n_seq_neurons, (512 - n_neurons)])
    neuron_labels = ['Ramping cells', 'Sequence cells', 'Other cells']
    plt.figure()
    plt.pie(neuron_counts, labels=neuron_labels, autopct=make_autopct(neuron_counts))
    plt.title(label)
    if save:
        plt.savefig(os.path.join(save_dir, label+'piechart.svg'))
    else:
        plt.show()


def make_venn_diagram(cell_nums_ramp, cell_nums_seq, n_neurons, save_dir, label, save=False):
    total_neuron_idx = np.arange(n_neurons)
    venn2(subsets=(len(cell_nums_ramp), len(cell_nums_seq), len(np.intersect1d(cell_nums_ramp, cell_nums_seq))), \
          set_labels=('Ramping', 'Sequence'))
    plt.gca().set_facecolor('skyblue')
    plt.gca().set_axis_on()
    plt.title(label)
    if save:
        plt.savefig(os.path.join(save_dir, label+'_venndiagram.svg'))
    else:
        plt.show()


def plot_sorted_averaged_resp(cell_nums, sorted_matrix, title, remove_nan=True, save_dir=None, save=False):
    """
    Plot sorted normalized average-response matrix. On y-axis, display where in the layer the cell is.
    Note: normalize whole range, not just absolute value
    Arguments:
    - cell_nums
    - sorted_matrix
    - title: str
    """

    len_delay = np.shape(sorted_matrix)[1]
    entropy, ts, sqi = sequentiality_analysis(sorted_matrix)
    if remove_nan:
        # Remove NaNs
        mask = np.all(np.isnan(sorted_matrix), axis=1)
        sorted_matrix = sorted_matrix[~mask]
        cell_nums = cell_nums[~mask]
    fig, ax = plt.subplots(figsize=(6, 9))
    cax = ax.imshow(sorted_matrix, cmap='jet')
    cbar = plt.colorbar(cax, ax=ax, label='Normalized Unit Activation')
    ax.set_aspect('auto')
    ax.set_yticks([0, len(cell_nums)])
    ax.set_yticklabels([1, f'{len(cell_nums)}'])
    ax.set_xticks(np.arange(len_delay, step=10))
    ax.set_xlabel('Time since delay onset')
    ax.set_ylabel('Unit #')
    ax.set_title(title + f' \n PE={entropy:.2f} \n TS={ts:.2f} \n SqI={sqi:.2f}')
    if save:
        plt.savefig(os.path.join(save_dir,  f'{title}.svg'))
    else:
        plt.show()


def plot_sorted_in_same_order(resp_a, resp_b, a_title, b_title, big_title, len_delay, n_neurons, save_dir, remove_nan=True, save=False):
    """
    Given response matrices a and b (plotted on the left and right, respectively),
    plot sorted_averaged_resp (between 0 and 1) for matrix a, then sort matrix b
    according the cell order that gives tiling pattern for matrix a
    Args:
    - resp_a, resp_b: arrays
    - a_title, b_title, big_title: strings
    - len_delay
    - n_neurons
    """
    segments_a = np.moveaxis(resp_a, 0, 1)
    segments_b = np.moveaxis(resp_b, 0, 1)
    unsorted_matrix_a = np.zeros((n_neurons, len_delay))
    unsorted_matrix_b = np.zeros((n_neurons, len_delay))
    sorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrix_b = np.zeros((n_neurons, len_delay))
    for i in range(len(segments_a)):  # at timestep i
        averages_a = np.mean(segments_a[i],
                             axis=0)  # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
        averages_b = np.mean(segments_b[i], axis=0)
        unsorted_matrix_a[:, i] = np.transpose(
            averages_a)  # goes into the i-th column of unsorted_matrix, each row is one neuron
        unsorted_matrix_b[:, i] = np.transpose(averages_b)
    normalized_matrix_a = (unsorted_matrix_a - np.min(unsorted_matrix_a, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_a, axis=1, keepdims=True)
    # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
    normalized_matrix_b = (unsorted_matrix_b - np.min(unsorted_matrix_b, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_b, axis=1, keepdims=True)
    max_indeces_a = np.argmax(normalized_matrix_a, axis=1)  # which time step does the maximum firing occur
    cell_nums_a = np.argsort(max_indeces_a)  # returns the order of cell number that should go into sorted_matrix
    for i, i_cell in enumerate(list(cell_nums_a)):
        sorted_matrix_a[i] = normalized_matrix_a[i_cell]
        sorted_matrix_b[i] = normalized_matrix_b[i_cell]  # sort b according to order in a

    if remove_nan:
        mask = np.logical_or(np.all(np.isnan(sorted_matrix_b), axis=1), np.all(np.isnan(sorted_matrix_b), axis=1))
        sorted_matrix_a = sorted_matrix_a[~mask]
        cell_nums_a = cell_nums_a[~mask]
        sorted_matrix_b = sorted_matrix_b[~mask]

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(7, 8))
    ax.imshow(sorted_matrix_a, cmap='jet')
    ax2.imshow(sorted_matrix_b, cmap='jet')
    ax.set_ylabel("Unit #")
    #ax.set_yticklabels(['1', f'{len(cell_nums_a)}'])
    #ax2.set_yticklabels(['1', f'{len(cell_nums_a)}'])
    ax.set_title(a_title)
    ax2.set_title(b_title)
    ax.set_yticks([0, len(cell_nums_a)])
    ax2.set_yticks([0, len(cell_nums_a)])
    ax.set_xticks(np.arange(len_delay, step=10))
    ax2.set_xticks(np.arange(len_delay, step=10))
    ax.set_xlabel('Time since delay onset')
    ax2.set_xlabel('Time since delay onset')
    fig.suptitle(big_title)
    ax.set_aspect('auto')
    ax2.set_aspect('auto')
    if save:
        plt.savefig(os.path.join(save_dir, f'sorted_in_same_order_{big_title}.svg'))
    else:
        plt.show()


def decode_sample_from_single_time(total_resp, total_stim, n_fold=5):
    """
    Returns:
    - accuracies: array of shape n_fold x len_delay
    - accuracies_shuff: unit-shuffled. array of shape  n_fold x len_delay
    """
    from sklearn.model_selection import KFold
    len_delay = np.shape(total_resp)[1]
    accuracies = np.zeros((n_fold, len_delay))
    accuracies_shuff = np.zeros((n_fold, len_delay))
    kf = KFold(n_splits=n_fold)
    segments = np.moveaxis(total_resp, 0, 1)
    for t in range(len_delay): # for each time step
        resp = stats.zscore(segments[t], axis=1)  # z-normalized
        resp_shuff = np.stack([np.random.permutation(x) for x in resp])
        i_split = 0
        for train_index, test_index in kf.split(total_resp): # for each fold
            r_train, r_test = resp[train_index], resp[test_index]
            s_train, s_test = total_stim[train_index], total_stim[test_index]

            clf = svm.SVC()
            clf.fit(r_train, s_train)
            s_test_pred = clf.predict(r_test)
            accuracies[i_split, t] = np.mean(s_test_pred == s_test)

            r_train_shuff, r_test_shuff = resp_shuff[train_index], resp_shuff[test_index]
            s_train_shuff, s_test_shuff = total_stim[train_index], total_stim[test_index]
            clf = svm.SVC()
            clf.fit(r_train_shuff, s_train_shuff)
            s_test_pred_shuff = clf.predict(r_test_shuff)
            accuracies_shuff[i_split, t] = np.mean(s_test_pred_shuff == s_test_shuff)
            i_split += 1
    return accuracies, accuracies_shuff


def plot_decode_sample_from_single_time(total_resp, total_stim, title, save_dir, n_fold=5, max_iter=100, save=False):
    """
    Arguments:
    - total_resp (eg. lstm, or first delay)
    - total_stim
    - title: str
    - max_iter: for LogisticRegression (default = 100)
    """
    accuracies, accuracies_shuff = decode_sample_from_single_time(total_resp, total_stim, n_fold=n_fold)
    len_delay = np.shape(total_resp)[1]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len_delay), np.mean(accuracies, axis=0), label='unshuffled', color=utils_linclab_plot.LINCLAB_COLS['green']) # TODO: green/purple for mem/nomem
    ax.fill_between(np.arange(len_delay), np.mean(accuracies, axis=0) - np.std(accuracies, axis=0), np.mean(accuracies, axis=0) + np.std(accuracies, axis=0), facecolor=
    utils_linclab_plot.LINCLAB_COLS['green'], alpha=0.5)
    ax.plot(np.arange(len_delay), np.mean(accuracies_shuff, axis=0), label='unit-shuffled', color=utils_linclab_plot.LINCLAB_COLS['grey'])
    ax.fill_between(np.arange(len_delay), np.mean(accuracies_shuff, axis=0) - np.std(accuracies_shuff, axis=0), np.mean(accuracies_shuff, axis=0) + np.std(accuracies_shuff, axis=0), facecolor=
    utils_linclab_plot.LINCLAB_COLS['grey'], alpha=0.5)
    ax.set(xlabel='Time since delay onset', ylabel='Stimulus decoding accuracy',
           title=title)
    ax.set_xticks(np.arange(len_delay, step=10))
    ax.legend()
    if save:
        plt.savefig(os.path.join(save_dir, f'decode_stim_{title}.svg'))
    else:
        plt.show()


def time_decode_lin_reg(delay_resp, len_delay, n_neurons, bin_size, title, save_dir, save=False):
    """
    Decode time with linear regression.
    :param delay_resp: n_episodes x len_delay x n_neurons
    :param len_delay: int
    :param n_neurons: int
    :param bin_size: int (1000)
    :param title: str
    :param plot: bool (default=False). Plot p_matrix as heatmap, with blue line indicating highest-probability decoded bin
    :return: p_matrix: len_delay (decoded) x len_delay (elapsed), each entry is probability of decoded time given resp at elapsed time
    :return: time_decode_error: mean absolute value of error-percentage
    :return: time_deocde_entropy: entropy of the probability matrix
    """
    global figpath
    clf = LinearRegression()
    epi_t = np.array(np.meshgrid(np.arange(0, bin_size), np.arange(len_delay))).T.reshape(-1, 2)
    np.random.shuffle(epi_t)  # random combination of episode number and time
    percent_train = 0.6
    epi_t_train = epi_t[:int(percent_train * len_delay * bin_size)]  # 0.6*40000 by 2
    epi_t_test = epi_t[int(percent_train * len_delay * bin_size):]
    r_train = np.zeros((len(epi_t_train), n_neurons))
    r_test = np.zeros((len(epi_t_test), n_neurons))
    for i in range(len(epi_t_train)):
        r_train[i] = delay_resp[epi_t_train[i, 0], epi_t_train[i, 1], :]
    for i in range(len(epi_t_test)):
        r_test[i] = delay_resp[epi_t_test[i, 0], epi_t_test[i, 1], :]
    t_train = np.squeeze(epi_t_train[:, 1])
    t_test = np.squeeze(epi_t_test[:, 1])

    reg = clf.fit(r_train, t_train)
    print("R2 on train: ", reg.score(r_train, t_train))
    t_test_pred = reg.predict(r_test)
    mean_pred = np.zeros(len_delay)
    std_pred = np.zeros(len_delay)
    for t_elapsed in range(len_delay):
        mean_pred[t_elapsed] = np.mean(t_test_pred[t_test == t_elapsed])
        std_pred[t_elapsed] = np.std(t_test_pred[t_test == t_elapsed])# 1 x len_delay

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x=t_test, y=t_test_pred,s=1)
    ax.set_xlabel('Time since delay onset')
    ax.set_ylabel('Decoded time')
    ax.plot(np.arange(len_delay), np.arange(len_delay),color='k')
    ax.plot(np.arange(len_delay), mean_pred, color=utils_linclab_plot.LINCLAB_COLS['blue'])
    ax.fill_between(np.arange(len_delay), mean_pred-std_pred,  mean_pred+std_pred, color='skyblue',alpha=0.4)
    ax.set_xticks([0, len_delay])
    ax.set_xticklabels(['0', str(len_delay)])
    ax.set_yticks([0, len_delay])
    ax.set_yticklabels(['0', str(len_delay)])
    ax.set_title(f'Decode Time {title}')
    ax.set_xlim((np.min(t_test_pred),np.max(t_test_pred)))
    ax.set_ylim((0,len_delay))
    if save:
        plt.savefig(os.path.join(save_dir, f'decode_time_linreg_{title}.svg'))
    else:
        plt.show()
    return t_test, t_test_pred

def single_cell_visualization(total_resp, binary_stim, cell_nums, type, save_dir):
    len_delay = np.shape(total_resp)[1]
    n_neurons = np.shape(total_resp)[2]
    print("Number of left trials: ", np.sum(binary_stim==1))
    print("Number of right trials: ", np.sum(binary_stim==0))

    assert len(total_resp) == len(binary_stim)
    assert all(elem in list(np.arange(n_neurons)) for elem in cell_nums)

    for i_neuron in cell_nums:
        #idx = np.random.permutation(np.min(np.sum(binary_stim==0), np.sum(binary_stim==1)))[:100]
        xl = total_resp[binary_stim == 0, :, i_neuron][-100:]
        xr = total_resp[binary_stim == 1, :, i_neuron][-100:]
        norm_xl = stats.zscore(xl, axis=1)
        norm_xr = stats.zscore(xr, axis=1)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(5, 8), sharex='all',
                                            gridspec_kw={'height_ratios': [2, 2, 1.5]})
        fig.suptitle(f'Unit #{i_neuron}')

        im = ax1.imshow(norm_xl, cmap='jet')
        ax1.set_aspect('auto')
        ax1.set_xticks(np.arange(len_delay, step=10))
        ax1.set_yticks([0, len(norm_xl)])
        ax1.set_yticklabels(['1', '100'])
        ax1.set_ylabel(f'Left trials')

        im2 = ax2.imshow(norm_xr, cmap='jet')
        ax2.set_aspect('auto')
        ax2.set_xticks(np.arange(len_delay, step=10))
        ax2.set_yticks([0, len(norm_xr)])
        ax2.set_yticklabels(['1', '100'])
        ax2.set_ylabel(f'Right trials')

        ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xl, axis=0), axis=0), label='Left', color=utils_linclab_plot.LINCLAB_COLS['yellow'])
        ax3.plot(np.arange(len_delay), stats.zscore(np.mean(xr, axis=0), axis=0), label='Right', color=utils_linclab_plot.LINCLAB_COLS['brown'])
        ax3.set_xlabel('Time since delay period onset')
        ax3.legend(loc='upper right', fontsize='medium')
        ax3.set_ylabel('Avg activation')
        # plt.show()
        if not os.path.exists(os.path.join(save_dir, 'single_unit_v_time')):
            os.mkdir(os.path.join(save_dir, 'single_unit_v_time'))
        if not os.path.exists(os.path.join(save_dir, 'single_unit_v_time', type)):
            os.mkdir(os.path.join(save_dir, 'single_unit_v_time', type))
        plt.savefig(os.path.join(save_dir, 'single_unit_v_time', type, f'{i_neuron}.svg'))


def time_decode(delay_resp, len_delay, n_neurons, bin_size, save_dir, title, plot=False, save=False):
    """
    Decode time with multiclass logistic regression.
    :param delay_resp: n_episodes x len_delay x n_neurons
    :param len_delay: int
    :param n_neurons: int
    :param bin_size: int
    :param title: str
    :param plot: bool (default=False). Plot p_matrix as heatmap, with blue line indicating highest-probability decoded bin
    :return: p_matrix: len_delay (decoded) x len_delay (elapsed), each entry is probability of decoded time given resp at elapsed time
    :return: time_decode_error: mean absolute value of error-percentage
    :return: time_deocde_entropy: entropy of the probability matrix
    """
    p_matrix = np.zeros((len_delay, len_delay))
    clf = LogisticRegression(multi_class='multinomial')
    epi_t = np.array(np.meshgrid(np.arange(0, bin_size), np.arange(len_delay))).T.reshape(-1, 2)
    np.random.shuffle(epi_t)  # random combination of episode number and time
    percent_train = 0.6
    epi_t_train = epi_t[:int(percent_train * len_delay * bin_size)]  # 0.6*40000 by 2
    epi_t_test = epi_t[int(percent_train * len_delay * bin_size):]
    r_train = np.zeros((len(epi_t_train), n_neurons))
    r_test = np.zeros((len(epi_t_test), n_neurons))
    for i in range(len(epi_t_train)):
        r_train[i] = delay_resp[epi_t_train[i, 0], epi_t_train[i, 1], :]
    for i in range(len(epi_t_test)):
        r_test[i] = delay_resp[epi_t_test[i, 0], epi_t_test[i, 1], :]
    t_train = np.squeeze(epi_t_train[:, 1])
    t_test = np.squeeze(epi_t_test[:, 1])

    clf.fit(r_train, t_train)
    for t_elapsed in range(len_delay):
        p_matrix[:, t_elapsed] = np.mean(clf.predict_proba(r_test[t_test == t_elapsed]), axis=0)  # 1 x len_delay
    decoded_time = np.argmax(p_matrix, axis=0)
    # time_decode_rmsep = np.sqrt(np.mean(((decoded_time - np.arange(len_delay)) / len_delay)**2))  # root mean squared error-percentage
    time_decode_error = np.mean(
        np.abs((decoded_time - np.arange(len_delay)) / len_delay))  # mean absolute error-percentage
    time_decode_entropy = np.mean(stats.entropy(p_matrix, axis=0))

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(title)
        cax = ax.imshow(p_matrix, cmap='hot')
        ax.set_xlabel('Time since delay onset')
        ax.set_ylabel('Decoded time')
        cbar = plt.colorbar(cax, ax=ax, label='p', aspect=10, shrink=0.5)
        ax.set_title(f'Accuracy={100 * (1 - time_decode_error):.2f}%')
        ax.plot(np.arange(len_delay), decoded_time)
        ax.set_xticks([0, len_delay])
        ax.set_xticklabels(['0', str(len_delay)])
        ax.set_yticks([0, len_delay])
        ax.set_yticklabels(['0', str(len_delay)])
        if save:
            plt.savefig(os.path.join(save_dir, f'decode_time_{title}.svg'))
        else:
            plt.show()

    return p_matrix, time_decode_error, time_decode_entropy


def plot_half_split_resp(total_resp, save_dir, split='random', save=False):
    n_total_episodes = np.shape(total_resp)[0]
    len_delay = np.shape(total_resp)[1]
    n_neurons = np.shape(total_resp)[2]

    if split == 'random':
        split_1_idx = np.random.choice(n_total_episodes, n_total_episodes//2, replace=False)
        ind = np.zeros(n_total_episodes, dtype=bool)
        ind[split_1_idx] = True
        rest = ~ind
        split_2_idx = np.arange(n_total_episodes)[rest]
        title_1 = "Random split 1st half"
        title_2 = "Random split 2nd half"
    elif split == 'odd-even':
        split_1_idx = np.arange(start=0, stop=n_total_episodes, step=2)
        split_2_idx = np.arange(start=1, stop=n_total_episodes+1, step=2)
        title_1 = "Odd trials"
        title_2 = "Even trials"

    resp_1 = total_resp[split_1_idx]
    resp_2 = total_resp[split_2_idx]

    plot_sorted_in_same_order(resp_1, resp_2, title_1, title_2, "", len_delay=len_delay, n_neurons=n_neurons, save_dir=save_dir, save=save)


def tuning_curve_dim_reduction(resp, mode, save_dir, title):
    # TODO: label clusters
    # tuning_curves = np.mean(resp, axis=0).T  # n_neurons x len_delay
    n_total_episodes = np.shape(resp)[0]
    len_delay = np.shape(resp)[1]
    n_neurons = np.shape(resp)[2]
    tuning_curves = np.reshape(resp, (n_neurons, n_total_episodes*len_delay))
    if mode == 'umap':
        embedding = umap.UMAP(n_neighbors=40).fit_transform(tuning_curves)
    elif mode == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)  # recommended n_components: 2
        embedding = tsne.fit_transform(tuning_curves)
    elif mode == 'pca':
        pca = PCA(n_components=10)  # recommended n_components: 10. Though raising it to 20 seems to not make a difference?
        embedding = pca.fit_transform(tuning_curves)

    plt.figure()
    plt.scatter(x=embedding[:, 0],
                y=embedding[:, 1], alpha=1)
    plt.title(f"{title}_{mode}")
    plt.show()
    # plt.savefig(os.path.join(save_dir,f'{title}_tuning_curve_{mode}.svg'))


def shuffle_activity(delay_resp):
    #shuffle LSTM activity within each episode for all neurons
    shuffled_delay_resp = np.empty(np.shape(delay_resp))
    n_episodes = np.shape(delay_resp)[0]
    len_delay = np.shape(delay_resp)[1]
    shift = np.random.randint(np.floor(len_delay*0.3), np.ceil(len_delay*0.7), size=n_episodes)
    for i_eps in range(n_episodes):
        shuffled_delay_resp[i_eps,:,:] = np.roll(np.squeeze(delay_resp[i_eps,:,:]), shift=shift[i_eps], axis=0)
    return shuffled_delay_resp


def shuffle_activity_single_neuron(delay_resp):
    #shuffle LSTM activity within each episode for a single neuron
    assert len(delay_resp.shape) == 2, "must only input resp for single neuron"
    shuffled_delay_resp = np.empty(np.shape(delay_resp))
    n_episodes = np.shape(delay_resp)[0]
    len_delay = np.shape(delay_resp)[1]
    shift = np.random.randint(np.floor(len_delay*0.3), np.ceil(len_delay*0.7), size=n_episodes)
    for i_eps in range(n_episodes):
        shuffled_delay_resp[i_eps,:] = np.roll(delay_resp[i_eps,:], shift=shift[i_eps])
    return shuffled_delay_resp


def shuffle_activity_single_neuron_varying_duration(delay_resp, stim_duration, return_nan=False):
    """
    :param delay_resp: n_episodes x len_delay. May contain 0 or NaN.
    :param stim_duration: n_episodes. Each element is the length (eg. 15, 30)
    :param return_nan: default=False, i.e. return shuffled response that contains 0
    :return: shuffled_delay_resp: n_episodes x len_delay. default return 0 for empty elements.
    """
    #shuffle LSTM activity within each episode for a single neuron. Trials may have different lengths, stored in stim_duration

    assert len(delay_resp.shape) == 2, "must only input resp for single neuron"
    shuffled_delay_resp = np.zeros_like(delay_resp)
    n_episodes = np.shape(delay_resp)[0]
    for i_eps in range(n_episodes):
        epi_resp = delay_resp[i_eps, :stim_duration[i_eps]]
        shift = np.random.randint(np.floor(stim_duration[i_eps]*0.3), np.ceil(stim_duration[i_eps]*0.7))
        shuffled_delay_resp[i_eps, :stim_duration[i_eps]] = np.roll(epi_resp, shift=shift)
    if return_nan:
        shuffled_delay_resp[shuffled_delay_resp == 0] = np.nan
    return shuffled_delay_resp

# ================= TO CHUCK =================================

def plot_dim_vs_delay_t(delay_resp, title, save_dir, n_trials=5, var_explained=0.9):
    """
    Plot dimension (by default explains 90% variance) of single-time activities vs elapsed time.
    :param delay_resp: n_episodes x len_delay x n_neurons. Use a single analysis bin (eg. 1000 episodes) rather than
    entire training session.
    :param n_trials: number of trials to average over
    :param var_explained: how much variance you want the dimension. Default = 0.9
    """
    len_delay = np.shape(delay_resp)[1]
    dim = np.zeros((n_trials, len_delay-1))
    epi_shuff = np.arange(int(len(delay_resp)))
    np.random.shuffle(epi_shuff)
    for i_trial in range(n_trials):
        episode = epi_shuff[i_trial]
        for t in range(1, len_delay):
            delay_resp_t = delay_resp[episode, :t+1, :]
            pca_model = PCA()
            pca_model.fit(delay_resp_t)
            cumsum = pca_model.explained_variance_ratio_.cumsum()
            dim[i_trial, t-1] = next(x[0] for x in enumerate(cumsum) if x[1] > var_explained)
    fig, ax0 = plt.subplots()
    ax0.plot(np.arange(len_delay-1), np.mean(dim, axis=0), color=utils_linclab_plot.LINCLAB_COLS['blue'])
    ax0.fill_between(np.arange(len_delay-1), np.mean(dim, axis=0) - np.std(dim, axis=0), np.mean(dim, axis=0) + np.std(dim, axis=0), color='skyblue')
    ax0.set_xlabel('Time since delay onset')
    ax0.set_ylabel('Cumulative Dimensionality')
    ax0.set_title(title)
    plt.show()
    # plt.savefig(os.path.join(save_dir + f'dim_v_delay_{title}.svg'))


def separate_vd_resp(resp, len_delay):
    '''
    :param resp: array of n_total_episodes x max(len_delays) x n_neurons
    :param len_delay: array of n_total_episodes
    :return: resp_dict: with keys = unique delay lengths, values = arrays storing resp corresponding to that delay length
    :return: counts: number of occurrence of each delay length
    '''
    len_delays, counts = np.unique(len_delay, return_counts=True)
    resp_dict = dict.fromkeys(len_delays)
    for ld in len_delays:  # for each unique delay length
        resp_dict[ld] = resp[len_delay == ld][:, :ld, :]
    return resp_dict, counts


def sequentiality_analysis(sorted_matrix):
    # Sequentiality index (Zhou 2020)
    # Arguments: sorted_matrix: n_neurons x len_delay
    # Return: peak entropy, temporal sparsity, sequentiality index
    len_delay = np.shape(sorted_matrix)[1]
    n_neurons = np.shape(sorted_matrix)[0]

    p_js = []  # to store p_j
    ts_ts = []  # to store TS(t)
    for t in range(len_delay):
        p_js.append(np.sum(np.argmax(sorted_matrix, axis=1) == t))  # number of units that peak at t
        r_i_t = sorted_matrix[:, t]
        r_i_t = r_i_t / np.nansum(r_i_t)
        ts_ts.append(np.nansum(-(r_i_t * np.log(r_i_t))) / np.log(n_neurons))  # TS(t)
    p_js = np.asarray(p_js) + 0.1  # add pseudocount to avoid log(0)
    ts_ts = np.asarray(ts_ts)
    peak_entropy = stats.entropy(p_js) / np.log(len_delay)
    temporal_sparsity = 1 - np.mean(ts_ts)

    sqi = np.sqrt(peak_entropy * temporal_sparsity)
    return peak_entropy, temporal_sparsity, sqi


def plot_sorted_vd(resp_dict, remove_nan=True):
    # Argument: resp_dict -- keys=length of delay, values = reps matrix
    # Sort each resp matrices according to the order of neurons of the first resp matrix, and plot as heat maps of n_neurons x len_delay
    # requirement: equal number of neurons in all resp matrices
    len_delays = list(resp_dict.keys())
    resp_matrices = list(resp_dict.values())
    resp_a = resp_matrices[0]
    len_delay = len_delays[0]
    n_neurons = np.shape(resp_a)[2]
    segments_a = np.moveaxis(resp_a, 0, 1)
    unsorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrix_a = np.zeros((n_neurons, len_delay))
    sorted_matrices = []
    for i in range(len_delay):  # at timestep i
        averages_a = np.mean(segments_a[i], axis=0)
        unsorted_matrix_a[:, i] = np.transpose(averages_a)
    normalized_matrix_a = (unsorted_matrix_a - np.min(unsorted_matrix_a, axis=1, keepdims=True)) / np.ptp(
        unsorted_matrix_a, axis=1, keepdims=True)
    max_indeces_a = np.argmax(normalized_matrix_a, axis=1)
    cell_nums_a = np.argsort(max_indeces_a)  # Get the cell order
    for i, i_cell in enumerate(list(cell_nums_a)):
        sorted_matrix_a[i] = normalized_matrix_a[i_cell]
    sorted_matrices.append(sorted_matrix_a)

    for resp in resp_matrices[1:]:  # for the rest of the response matrices
        ld = np.shape(resp)[1]
        segments = np.moveaxis(resp, 0, 1)
        unsorted_matrix = np.zeros((n_neurons, ld))
        sorted_matrix = np.zeros((n_neurons, ld))
        for i in range(ld):  # at timestep i
            # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
            averages = np.mean(segments[i], axis=0)
            # goes into the i-th column of unsorted_matrix, each row is one neuron
            unsorted_matrix[:, i] = np.transpose(averages)
        # 0=minimum response of this neuron over time, 1=maximum response of this neuro over time
        normalized_matrix = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) / np.ptp(unsorted_matrix,
                                                                                                        axis=1,
                                                                                                        keepdims=True)
        for i, i_cell in enumerate(list(cell_nums_a)):
            sorted_matrix[i] = normalized_matrix[i_cell]  # SORT ACCORDING TO RESP A'S ORDER
        sorted_matrices.append(sorted_matrix)

    if remove_nan:
        mask = np.logical_or(np.all(np.isnan(sorted_matrices[0]), axis=1), np.all(np.isnan(sorted_matrices[1]), axis=1), np.all(np.isnan(sorted_matrices[2]), axis=1))
        sorted_matrices[0] = sorted_matrices[0][~mask]
        sorted_matrices[1] = sorted_matrices[1][~mask]
        sorted_matrices[2] = sorted_matrices[2][~mask]

    from mpl_toolkits.axes_grid1 import AxesGrid
    fig = plt.figure(figsize=(6, 8))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, len(sorted_matrices)),
                    axes_pad=0.05,
                    share_all=False,
                    label_mode="L",
                    cbar_location="right",
                    cbar_size="20%",
                    cbar_mode="single",
                    )
    for sm, ax in zip(sorted_matrices, grid):
        print(len(sm))
        im = ax.imshow(sm, vmin=0, vmax=1, cmap='jet')
        ax.set_xticks(np.arange(10, np.shape(sm)[1] + 10, 10))
        ax.set_aspect(0.4)  # the smaller this number, the wider the plot. 1 means no horizontal stretch.
        ax.set_xlabel('Time')
        ax.set_ylabel("Unit #")
        ax.set_yticks([0, len(sm)])
        ax.set_yticklabels(['1', f'{len(sm)}'])
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
        cax.toggle_label(True)
    plt.show()


def split_train_and_test(percent_train, total_resp, total_stim, seed):
    """
    Split a neural activity matrix of shape n_stimuli x n_features into training
    (contains percent_train of data) and testing sets.
    Arguments:
    - percent_train (a number between 0 and 1)
    - total_resp (np.array of shape n_stimuli x n_features)
    - total_stim (np.array of shape n_stimuli x 1, each entry is 0 or 1)
    - seed
    Returns:
    - resp_train
    - resp_test
    - stimuli_train
    - stimuli_test
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_stimuli = total_resp.shape[0]
    n_train = int(percent_train * n_stimuli)  # use 60% of all data for training set
    ishuffle = torch.randperm(n_stimuli)
    itrain = ishuffle[:n_train]  # indices of data samples to include in training set
    itest = ishuffle[n_train:]  # indices of data samples to include in testing set
    stimuli_test = total_stim[itest]
    resp_test = total_resp[itest]
    stimuli_train = total_stim[itrain]
    resp_train = total_resp[itrain]
    return resp_train, resp_test, stimuli_train, stimuli_test


def sort_resp(total_resp, norm=True):
    """
    Average the responses across episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and sort cells by when their maximum response happens.
    returns: cell_nums, sorted_matrix
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

    return cell_nums, sorted_matrix
