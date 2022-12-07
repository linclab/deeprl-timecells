import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn import svm
import numpy as np
from linclab_utils import plot_utils


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


def plot_decode_sample_from_single_time(total_resp, total_stim, title, n_fold=5, max_iter=100):
    """
    Arguments:
    - total_resp (eg. lstm, or first delay)
    - total_stim
    - title: str
    - max_iter: for LogisticRegression (default = 100)
    """
    global figpath
    accuracies, accuracies_shuff = decode_sample_from_single_time(total_resp, total_stim, n_fold=n_fold)
    len_delay = np.shape(total_resp)[1]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len_delay), np.mean(accuracies, axis=0), label='unshuffled', color=plot_utils.LINCLAB_COLS['green']) # TODO: green/purple for mem/nomem
    ax.fill_between(np.arange(len_delay), np.mean(accuracies, axis=0) - np.std(accuracies, axis=0), np.mean(accuracies, axis=0) + np.std(accuracies, axis=0), facecolor=plot_utils.LINCLAB_COLS['green'], alpha=0.5)
    ax.plot(np.arange(len_delay), np.mean(accuracies_shuff, axis=0), label='unit-shuffled', color=plot_utils.LINCLAB_COLS['grey'])
    ax.fill_between(np.arange(len_delay), np.mean(accuracies_shuff, axis=0) - np.std(accuracies_shuff, axis=0), np.mean(accuracies_shuff, axis=0) + np.std(accuracies_shuff, axis=0), facecolor=plot_utils.LINCLAB_COLS['grey'], alpha=0.5)
    ax.set(xlabel='Time since delay onset', ylabel='Stimulus decoding accuracy',
           title=title)
    ax.set_xticks(np.arange(len_delay, step=10))
    ax.legend()
    plt.show()
    #plt.savefig(figpath + f'/decode_stim_{title}.png')


def time_decode(delay_resp, len_delay, n_neurons, bin_size, title, plot=False):
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
    global figpath
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
        plt.show()
        #plt.savefig(figpath + f'/decode_time_{title}.png')

    return p_matrix, time_decode_error, time_decode_entropy
