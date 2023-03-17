import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import svm
import os
from utils_analysis import make_piechart, plot_sorted_averaged_resp, plot_sorted_in_same_order, time_decode
from utils_time_ramp import *
import numpy as np


def compare_correct_vs_incorrect(resp, stim_all, correct_trial, title, save_dir, save=False, analysis="population",
                                 resp2=None):
    corr_resp = resp[correct_trial == 1, :, :]
    incorr_resp = resp[correct_trial == 0, :, :]
    print("Number of correct episodes: ", np.shape(corr_resp)[0])
    print("Number of incorrect episodes: ", np.shape(incorr_resp)[0])
    # n_episodes = np.shape(incorr_resp)[0]
    # idx_episodes = np.random.choice(np.shape(corr_resp)[0], size=(n_episodes,1), replace=False)
    # corr_resp = np.squeeze(corr_resp[idx_episodes, :, :])
    # stim = np.zeros((n_episodes, 2))
    # stim[:,0] = np.squeeze(stim_all[idx_episodes, 0])
    # stim[:,1] = np.squeeze(stim_all[correct_trial==0, 1])

    if analysis == "decoding":
        assert resp2 is not None, 'Must provide resp2'
        # time_decode_all_stim_len(corr_resp, stim[:,0], title="Correct trials")
        # time_decode_all_stim_len(incorr_resp, stim[:,1], title="Incorrect trials")
        # time_decode(corr_resp, 40, 512, 1000, "Correct trials", plot=True)
        # time_decode(incorr_resp, 40, 512, 1000, "Incorrect trials", plot=True)
        # time_cross_decode(corr_resp, incorr_resp)

        # aggregate stim_1 and stim_2 activity
        corr_resp2 = resp2[correct_trial == 1, :, :]
        incorr_resp2 = resp2[correct_trial == 0, :, :]
        # idx_episodes = np.random.choice(np.shape(corr_resp2)[0], size=(n_episodes,1), replace=False)
        # corr_resp2 = np.squeeze(corr_resp2[idx_episodes, :, :])
        corr_resp = np.concatenate((corr_resp, corr_resp2), axis=0)
        incorr_resp = np.concatenate((incorr_resp, incorr_resp2), axis=0)

        n_split = 5
        duration = np.shape(corr_resp)[1]
        decoded_t_corr = np.zeros((n_split, duration))
        decoded_t_incorr = np.zeros((n_split, duration))
        kf = KFold(n_splits=n_split, shuffle=True)
        i_split = 0
        for train_idx, test_idx in kf.split(np.arange(np.shape(corr_resp)[0])):
            # print(train_idx)
            corr_train, corr_test = corr_resp[train_idx, :, :], corr_resp[test_idx, :, :]
            p_matrix_corr = \
            time_cross_decode(corr_train, corr_test, title="", bin_size=100, plot=False, save_dir=save_dir, save=save)[
                0]
            p_matrix_incorr = \
            time_cross_decode(corr_train, incorr_resp, title="", bin_size=100, plot=False, save_dir=save_dir,
                              save=save)[0]
            decoded_t_corr[i_split, :] = np.argmax(p_matrix_corr, axis=0)
            decoded_t_incorr[i_split, :] = np.argmax(p_matrix_incorr, axis=0)
            i_split += 1
            print(str(i_split) + " iteration has completed.")

        max_stim_len = 40
        fig, ax = plt.subplots()
        ax.plot(np.arange(max_stim_len), np.arange(max_stim_len), '--', color='gray', alpha=0.7,
                label="Decoded = actual time")
        ax.set_xlim([0, max_stim_len])
        ax.set_ylim([0, max_stim_len])
        ax.axis("on")
        ax.plot(np.arange(max_stim_len), np.mean(decoded_t_corr, axis=0), linewidth=3, label="Decode on correct trials")
        ax.plot(np.arange(max_stim_len), np.mean(decoded_t_incorr, axis=0), linewidth=3,
                label="Decode on incorrect trials")
        ax.fill_between(np.arange(max_stim_len), np.mean(decoded_t_corr, axis=0) + np.std(decoded_t_corr, axis=0),
                        np.mean(decoded_t_corr, axis=0) - np.std(decoded_t_corr, axis=0), alpha=0.3)
        ax.fill_between(np.arange(max_stim_len), np.mean(decoded_t_incorr, axis=0) + np.std(decoded_t_incorr, axis=0),
                        np.mean(decoded_t_incorr, axis=0) - np.std(decoded_t_incorr, axis=0), alpha=0.3)
        ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_xlabel("Time")
        ax.set_ylabel("Decoded Time")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        ax.legend(frameon=False)
        # ax.set_title("Accuracy = "+str(np.mean(accuracy*100))+"%")
        # ax.set_title(title)
        if save:
            plt.savefig(os.path.join(save_dir, f'compare_corr_vs_incorr_{title}.svg'))
        else:
            plt.show()

    elif analysis == "single_unit":
        single_cell_temporal_tuning(stim_all, corr_resp, incorr_resp, save_dir, compare_correct=True)
        # single_cell_temporal_tuning(stim, corr_resp, incorr_resp, save_dir)

    elif analysis == "population":
        assert resp2 is not None, 'Must provide resp2'
        corr_resp2 = resp2[correct_trial == 1, :, :]
        incorr_resp2 = resp2[correct_trial == 0, :, :]
        n_total_episodes = np.shape(corr_resp2)[0]
        idx_episodes = np.random.choice(np.shape(corr_resp2)[0], size=(n_total_episodes, 1), replace=False)
        corr_resp2 = np.squeeze(corr_resp2[idx_episodes, :, :])
        corr_resp = np.concatenate((corr_resp, corr_resp2), axis=0)
        incorr_resp = np.concatenate((incorr_resp, incorr_resp2), axis=0)
        plot_sorted_in_same_order(corr_resp, incorr_resp, 'Correct trials', 'Incorrect trials', big_title="",
                                  len_delay=40, n_neurons=512, save_dir=save_dir, save=save)


def time_decode_all_stim_len(stim_resp, stim, save_dir, title, save=False, bin_size=1000):
    """
    Given the population activity at each time, decode the time elasped since stimulus / delay onset
    """
    n_neurons = np.shape(stim_resp)[-1]
    all_stim_len = np.unique(stim)
    all_stim_len.astype(int)
    max_stim_len = np.max(stim)
    accuracy = np.zeros(7)

    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.plot(np.arange(max_stim_len), np.arange(max_stim_len), '--', label="correct prediction")
    ax.set_xlim([0, max_stim_len])
    ax.set_ylim([0, max_stim_len])
    ax.axis("on")

    for i_len, stim_len in enumerate(all_stim_len):
        stim_len = int(stim_len)
        resp = stim_resp[(stim == stim_len) * 1, :int(stim_len), :]
        p_matrix = time_decode(resp, int(stim_len), n_neurons, bin_size, save_dir, title)[0]
        decoded_time = np.argmax(p_matrix, axis=0)
        accuracy[i_len] = np.mean(decoded_time == np.arange(stim_len))
        ax.plot(np.arange(stim_len), decoded_time, alpha=0.5, label=str(stim_len) + " time steps")

    ax.set_xticks(all_stim_len)
    ax.set_yticks(all_stim_len)
    ax.set_xlabel("Time")
    ax.set_ylabel("Decoded Time")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("Accuracy = " + str(np.mean(accuracy * 100)) + "%")
    if save:
        plt.savefig(os.path.join(save_dir, 'time_decode_all_stim_len.svg'))
    else:
        plt.show()


def time_cross_decode(resp_corr, resp_incorr, title, save_dir, bin_size=340, plot=True, save=False):
    len_delay = np.shape(resp_corr)[1]
    n_neurons = np.shape(resp_incorr)[2]
    p_matrix = np.zeros((len_delay, len_delay))
    clf = LogisticRegression(multi_class='multinomial')
    epi_t_corr = np.array(np.meshgrid(np.arange(0, bin_size), np.arange(len_delay))).T.reshape(-1, 2)
    epi_t_incorr = np.array(np.meshgrid(np.arange(0, bin_size), np.arange(len_delay))).T.reshape(-1, 2)
    np.random.shuffle(epi_t_corr)  # random combination of episode number and time
    np.random.shuffle(epi_t_incorr)
    percent_train = 0.7
    # epi_t_train = epi_t_corr[:int(percent_train * len_delay * bin_size)]  # 0.6*40000 by 2
    # epi_t_test = epi_t_incorr[int(percent_train * len_delay * bin_size):]
    epi_t_train = epi_t_corr
    epi_t_test = epi_t_incorr
    r_train = np.zeros((len(epi_t_train), n_neurons))
    r_test = np.zeros((len(epi_t_test), n_neurons))

    for i in range(len(epi_t_train)):
        r_train[i] = resp_corr[epi_t_train[i, 0], epi_t_train[i, 1], :]
    for i in range(len(epi_t_test)):
            r_test[i] = resp_incorr[epi_t_test[i, 0], epi_t_test[i, 1], :]
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
        if save:
            plt.savefig(save_dir + f'/time_cross_decode_{title}.svg')
        else:
            plt.show()

    return p_matrix, time_decode_error, time_decode_entropy


def id_print_acc(stim, action_hist):
    print("Accuracy: ", np.sum(np.equal(action_hist == 1, np.greater(stim[:, 0], stim[:, 1])) * 1) / np.shape(stim)[0])


def plot_training_accuracy(stim, action_hist, title, save_dir, window_size=1000, base_episode=0, save=False):
    X = np.arange(np.shape(action_hist)[0]) + base_episode
    accuracy = np.equal(action_hist == 1, np.greater(stim[:, 0], stim[:, 1])) * 1
    accuracy = bin_rewards(accuracy, window_size)
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.plot(X, accuracy)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Accuracy")
    if save:
        plt.savefig(os.path.join(save_dir, f'{title}_training_accuracy.svg'))
    else:
        plt.show()


def plot_performance(stim, action_hist, title, save_dir, fig_type="matrix", save=False):
    accuracy = np.equal(action_hist == 1, np.greater(stim[:, 0], stim[:, 1])) * 1
    stim_set = np.sort(np.unique(stim))
    num_stim = np.max(np.shape(stim_set))
    if fig_type == "matrix":
        acc_matrix = np.zeros((num_stim, num_stim))
        for stim1_idx in range(num_stim):
            for stim2_idx in range(num_stim):
                if not stim1_idx == stim2_idx:
                    episode_idx = np.logical_and(stim[:, 0] == stim_set[stim1_idx], stim[:, 1] == stim_set[stim2_idx])
                    acc_matrix[stim1_idx, stim2_idx] = np.mean(accuracy[episode_idx])
        fig, ax = plt.subplots()
        pos = ax.imshow(acc_matrix)
        ax.set_xlabel("Stimulus 2")
        ax.set_ylabel("Stimulus 1")
        ax.set_title("Task performance")
        ax.set_xticks(list(range(num_stim)))
        ax.set_yticks(list(range(num_stim)))
        ax.set_xticklabels(stim_set)
        ax.set_yticklabels(stim_set)
        fig.colorbar(pos, ax=ax)
        if save:
            plt.savefig(os.path.join(save_dir, f'{title}_performance_matrix.svg'))
        else:
            plt.show()
    elif fig_type == "curve":
        duration_diff = [5, 10, 15, 20, 25, 30]
        acc_stim1_longer = np.zeros(len(duration_diff))
        acc_stim2_longer = np.zeros(len(duration_diff))
        for i in range(len(duration_diff)):
            acc_stim1_longer[i] = np.mean(accuracy[stim[:, 0] - stim[:, 1] == duration_diff[i]])
            acc_stim2_longer[i] = np.mean(accuracy[stim[:, 1] - stim[:, 0] == duration_diff[i]])
        fig, ax = plt.subplots()
        ax.plot(duration_diff, acc_stim1_longer, label="stim1 - stim2", marker='o')
        ax.plot(duration_diff, acc_stim2_longer, label="stim2 - stim1", marker='o')
        ax.set_xlabel("Difference in stimulus duration")
        ax.set_ylabel("Accuracy")
        ax.set_title("Task performance")
        ax.legend(frameon=False)
        if save:
            plt.savefig(os.path.join(save_dir, f'{title}_performance_curve.svg'))
        else:
            plt.show()


def plot_time_cell_sorted_same_order(stim, stim1_resp, stim2_resp, save_dir, save=False):
    stim_set = np.sort(np.unique(stim))
    num_neurons = np.shape(stim1_resp)[-1]
    for stim_len in stim_set:
        stim1_activities = stim1_resp[stim[:, 0] == stim_len, :stim_len, :]
        stim2_activities = stim2_resp[stim[:, 0] == stim_len, :stim_len, :]
        plot_sorted_in_same_order(stim1_activities, stim2_activities, 'Stim 1', 'Stim 2',
                                  big_title=str(stim_len) + "_steps", len_delay=stim_len, n_neurons=num_neurons,
                                  save_dir=save_dir, save=save)


def single_cell_temporal_tuning(stim, stim1_resp, stim2_resp, save_dir, compare_correct=False):
    stim_set = np.sort(np.unique(stim))
    stim_set.astype(int)
    num_stim = np.max(np.shape(stim_set))
    if not os.path.exists(os.path.join(save_dir, 'single_cell_temporal_tuning')):
        os.mkdir(os.path.join(save_dir, 'single_cell_temporal_tuning'))
    save_dir = os.path.join(save_dir, 'single_cell_temporal_tuning')

    if not compare_correct:
        for unit in range(np.shape(stim1_resp)[-1]):
            recep_field = np.zeros((num_stim, int(np.max(stim_set)), 2))
            recep_field[:, :, :] = np.nan
            for stim_idx in range(num_stim):
                stim_len = int(stim_set[stim_idx])
                episode_idx_1 = stim[:, 0] == stim_len
                episode_idx_2 = stim[:, 1] == stim_len
                recep_field[stim_idx, :stim_len, 0] = np.mean(np.squeeze(stim1_resp[episode_idx_1, :stim_len, unit]),
                                                              axis=0)
                recep_field[stim_idx, :stim_len, 1] = np.mean(np.squeeze(stim2_resp[episode_idx_2, :stim_len, unit]),
                                                              axis=0)
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.imshow(recep_field[:, :, 0], cmap='jet')
            ax2.imshow(recep_field[:, :, 1], cmap='jet')
            ax1.set_title('Correct trials')
            ax2.set_title('Incorrect trials')
            ax1.set_xticks(stim_set)
            ax2.set_xticks(stim_set)
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax1.set_xlabel('Stimulus length')
            ax2.set_xlabel('Stimulus length')
            fig.suptitle("Single Unit Temporal Tuning")
            ax1.set_aspect('auto')
            ax2.set_aspect('auto')
            plt.savefig(os.path.join(save_dir, f'{str(unit)}.svg'))


def plot_postlesion_performance(accuracy, compare="lesion type"):
    stim_set = [10, 15, 20, 25, 30, 35, 40]
    lesion_num = np.arange(13) * 10
    lesions = ["Random lesion", "Ramp lesion", "Seq lesion"]
    save_dir = 'data/analysis/lesion/'
    if compare == "lesion type":
        for i_stim, len_stim in enumerate(stim_set):
            fig, ax = plt.subplots()
            for i_lesion_type, lesion_type in enumerate(lesions):
                ax.plot(lesion_num, accuracy[i_stim, i_lesion_type, :], label=lesion_type)
            ax.set_xlabel("Number of units lesioned")
            ax.set_ylabel("Performance")
            ax.set_title("Lesion experiment")
            ax.legend(frameon=False)
            plt.savefig(save_dir + 'stim_' + str(len_stim) + '.svg')
    elif compare == "stimulus length":
        for i_lesion_type, lesion_type in enumerate(lesions):
            fig, ax = plt.subplots()
            for i_stim, len_stim in enumerate(stim_set):
                ax.plot(lesion_num, accuracy[i_stim, i_lesion_type, :], label="Stim1=" + str(len_stim))
            ax.set_xlabel("Number of units lesioned")
            ax.set_ylabel("Performance")
            ax.set_title("Lesion experiment")
            ax.legend(frameon=False)
            plt.savefig(save_dir + lesion_type + '.svg')


def retiming(stim, stim1_resp, stim2_resp, save_dir, verbose=False):
    '''
    To see whether there is experience-dependent retiming of single units.
    We select the episodes where the second stimulus is of a particular length (e.g.stim2=10),
    and compare the receptive field of single units during the second stimulus presentation,
    after they have gone through a first stimulus of different lengths.
    '''
    stim_set = np.unique(stim)
    n_neurons = np.shape(stim1_resp)[-1]
    if not os.path.exists(os.path.join(save_dir, 'retiming')):
        os.mkdir(os.path.join(save_dir, 'retiming'))
    save_dir = os.path.join(save_dir, 'retiming')
    for i_neuron in range(n_neurons):
        for i_stim2, stim2 in enumerate(stim_set):
            if not os.path.exists(os.path.join(save_dir, "Stim2_" + str(stim2))):
                os.mkdir(os.path.join(save_dir, "Stim2_" + str(stim2)))
            recep_field = np.zeros((len(stim_set) - 1, int(stim2)))
            possible_stim1 = np.delete(stim_set, i_stim2)
            for i_stim1, stim1 in enumerate(possible_stim1):
                i_episode = np.logical_and(stim[:, 0] == stim1, stim[:, 1] == stim2)
                recep_field[i_stim1, :] = np.mean(stim2_resp[i_episode, :stim2, i_neuron], axis=0)
            fig, ax = plt.subplots()
            cax = ax.imshow(recep_field, cmap='jet')
            fig.colorbar(cax, ax=ax)
            ax.set_yticks(np.arange(len(possible_stim1)))
            ax.set_yticklabels(possible_stim1)
            ax.set_ylabel("Stimulus 1")
            # ax.set_xlabel("Stimulus 1")
            ax.set_title("Stim2=" + str(stim2) + ", Unit " + str(i_neuron))
            plt.savefig(os.path.join(save_dir, "Stim2_" + str(stim2), "Unit_" + str(i_neuron) + ".svg"))
        if verbose:
            print("Unit", str(i_neuron), "finished.")


def linear_readout(stim, stim1_resp, stim2_resp, save_dir):
    '''
    Calculate the single unit activity for every pair of (stim1_len, stim2_len).
    The output is a (7,7) matrix representing the activity level for each of the 49
    combinations of stimulus length.
    '''
    stim_set = np.unique(stim)
    n_neurons = np.shape(stim1_resp)[-1]
    if not os.path.exists(os.path.join(save_dir, 'linear_readout')):
        os.mkdir(os.path.join(save_dir, 'linear_readout'))
    save_dir = os.path.join(save_dir, 'linear_readout')
    for i_neuron in range(n_neurons):
        readout = np.zeros((len(stim_set), len(stim_set)))
        for i_stim1, stim1 in enumerate(stim_set):
            for i_stim2, stim2 in enumerate(stim_set):
                i_episode = np.logical_and(stim[:, 0] == stim1, stim[:, 1] == stim2)
                readout[i_stim1, i_stim2] = np.mean(stim2_resp[i_episode, stim2 - 1, i_neuron])
        fig, ax = plt.subplots()
        cax = ax.imshow(readout, cmap='jet')
        fig.colorbar(cax, ax=ax)
        ax.set_yticks(np.arange(len(stim_set)))
        ax.set_xticks(np.arange(len(stim_set)))
        ax.set_yticklabels(stim_set)
        ax.set_xticklabels(stim_set)
        ax.set_ylabel("Stimulus 1")
        ax.set_xlabel("Stimulus 2")
        ax.set_title("Unit " + str(i_neuron))
        plt.savefig(os.path.join(save_dir, "Unit_" + str(i_neuron) + ".svg"))
        plt.close()


def decoding(stim, resp, save_dir, n_fold=10, save=False):
    n_target = 4
    kf = KFold(n_splits=n_fold)
    accuracies = np.zeros((n_target, n_fold))
    target = np.vstack((stim[:, 0], stim[:, 1], stim[:, 0] > stim[:, 1], stim[:, 0] - stim[:, 1]))
    for i in range(n_target):
        i_split = 0
        for train_index, test_index in kf.split(resp):  # for each fold
            r_train, r_test = resp[train_index, :], resp[test_index, :]
            # s_train, s_test = stim[train_index,1], stim[test_index,1]   # for first stimulus
            s_train, s_test = target[i, train_index], target[i, test_index]
            clf = svm.SVC()
            clf.fit(r_train, s_train)
            s_test_pred = clf.predict(r_test)
            accuracies[i, i_split] = np.mean(s_test_pred == s_test)
            i_split += 1
    fig, ax = plt.subplots()
    ax.bar(np.arange(n_target), np.mean(accuracies, axis=1), width=0.6)
    ax.errorbar(np.arange(n_target), np.mean(accuracies, axis=1), yerr=np.std(accuracies, axis=1), fmt='None',
                ecolor='dimgray')
    ax.set_xticks(np.arange(n_target))
    ax.set_xticklabels(['Stim1', 'Stim2', 'Stim1>Stim2', 'Stim1-Stim2'])
    ax.set_xlabel("Decoding target")
    ax.set_ylabel("Accuracy")
    if save:
        plt.savefig(os.path.join(save_dir, 'stim_decoding'))
        np.savez_compressed(os.path.join(save_dir, 'stim_decoding_accuracy.npz'), accuracies=accuracies)
    else:
        plt.show()


def plot_decoding_accuracy(acc, acc_shuffle, n_target=4):
    fig, ax = plt.subplots()
    ax.bar(np.arange(n_target), np.mean(acc, axis=1), width=0.6)
    ax.errorbar(np.arange(n_target), np.mean(acc, axis=1), yerr=np.std(acc, axis=1), fmt='None', ecolor='dimgray')
    ax.set_xticks(np.arange(n_target))
    ax.set_xticklabels(['Stim1', 'Stim2', 'Stim1>Stim2', 'Stim1-Stim2'])
    ax.set_xlabel("Decoding target")
    ax.set_ylabel("Accuracy")
    plt.show()


def manifold(stim, resp, save_dir, save=False):
    pca = PCA(n_components=5)
    pc = pca.fit_transform(resp)
    tsne = TSNE(n_components=2, init='pca')
    embeddings = tsne.fit_transform(pc)
    fig, ax = plt.subplots()
    ax.scatter(embeddings[stim[:, 0] < stim[:, 1], 0], embeddings[stim[:, 0] < stim[:, 1], 1],
               color='lightskyblue', alpha=0.7, s=20, label="Stim1 < Stim2")
    ax.scatter(embeddings[stim[:, 0] > stim[:, 1], 0], embeddings[stim[:, 0] > stim[:, 1], 1],
               color='lightcoral', alpha=0.7, s=20, label="Stim1 > Stim2")
    ax.legend()
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("t-SNE Results")
    if save:
        plt.savefig(os.path.join(save_dir, 'tsne.svg'))
    else:
        plt.show()


# ======== TODO: clean this up ======
def plot_training_performance(correct_trial, save_dir, n_seeds=5, save=False):
    n_total_epi = np.shape(correct_trial)[0]
    window_size = int(n_total_epi // 10)
    accuracies = np.zeros((n_seeds, n_total_epi))
    i_seed = 0
    for directory in os.listdir(parent_dir):
        if "lstm" in directory:
            directory = os.path.join(parent_dir, directory)
            file = directory + "/data.npz"
            data = np.load(file)
            accuracies[i_seed, :] = bin_rewards(correct_trial[:n_total_epi], window_size)
            i_seed += 1

    mean_acc = np.mean(accuracies, axis=0)
    mean_acc[mean_acc < 0.48] = 0.5
    fig, ax = plt.subplots()
    ax.hlines(y=1.0, xmin=0, xmax=n_total_epi, colors="gray", linestyles="--")
    ax.plot(np.arange(n_total_epi), mean_acc, linewidth=4)
    ax.fill_between(np.arange(n_total_epi), mean_acc + np.std(accuracies, axis=0) / np.sqrt(4),
                    mean_acc - np.std(accuracies, axis=0) / np.sqrt(4), alpha=0.3)
    ax.set_xticks([0, 50000, 100000, 150000])
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("% Correct response")
    if save:
        plt.savefig(os.path.join(save_dir, 'training_performance_seeds.svg'))
    else:
        plt.show()


# --------------------------------------------------------------------------------------------------------

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards) + 1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode - 1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode - 1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards


def plot_reward_hist(reward_hist, window=2000):
    num_episodes = np.shape(reward_hist)[0]
    ideal_rewards = 100 - 19 * 0.1  # calculate ideal reward
    ideal_rewards = np.ones(num_episodes) * ideal_rewards
    epi_rewards = np.sum(reward_hist, axis=1)  # calculate actual rewards
    avg_rewards = bin_rewards(epi_rewards, window)
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_episodes), avg_rewards, label="Actual")
    ax.plot(np.arange(num_episodes), ideal_rewards, label="Ideal")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Episode reward")
    ax.legend(frameon=False)
    plt.show()


def plot_incorrect_go(action_hist, window=1000):
    num_episodes = np.shape(action_hist)[0]
    num_wrong_act = np.sum(action_hist[:, :20] == 1, axis=1)
    num_wrong_act = bin_rewards(num_wrong_act, window)
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_episodes), num_wrong_act)
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Number of Incorrect 'Touch'")
    plt.show()


def plot_reaction_time(reward_hist, window=1000):
    num_episodes = np.shape(reward_hist)[0]
    reward_hist = reward_hist[:, 20:]
    react_time = np.sum(reward_hist == -0.1, axis=1)
    react_time = bin_rewards(react_time, window)
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_episodes), react_time)
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Reaction Time")
    plt.show()
