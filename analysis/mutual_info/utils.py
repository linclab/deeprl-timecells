from logging import info
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.cbook as cbook
from numpy import nan
from scipy import stats
from sklearn import svm
np.seterr(invalid='ignore')

import itertools
from matplotlib.cbook import _reshape_2D
import matplotlib.pyplot as plt
import numpy as np

# Function adapted from matplotlib.cbook
def my_boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                     autorange=False, percents=[25, 75]):

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # median
        med = np.percentile(x, 50)
        ## Altered line
        q1, q3 = np.percentile(x, (percents[0], percents[1]))

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats
# =====================================================

def compare_mutual_info(path):
    I = np.load(path)
    unrmd, posrmd, timermd = I['I_unsfl_unrmd'], I['I_unsfl_posrmd'], I['I_unsfl_timermd']
    #max_value = np.max(np.hstack((unrmd, posrmd, timermd)))+0.05
    max_value = 0.3

    fig, axs = plt.subplots()
    axs.scatter(unrmd, posrmd, c='rebeccapurple', alpha=0.4)
    axs.plot(np.linspace(0, max_value), np.linspace(0, max_value), linestyle='dashed', c='dimgray')
    axs.set_xlabel("T x Y")
    axs.set_ylabel("T x R(Y)")
    axs.set_xlim([0, max_value])
    axs.set_ylim([0, max_value])
    plt.show()

    fig, axs = plt.subplots()
    axs.scatter(unrmd, timermd, c='rebeccapurple', alpha=0.4)
    axs.plot(np.linspace(0, max_value), np.linspace(0, max_value), linestyle='dashed', c='dimgray')
    axs.set_xlabel("T x Y")
    axs.set_ylabel("R(T) x Y")
    axs.set_xlim([0, max_value])
    axs.set_ylim([0, max_value])
    plt.show()



def construct_ratemap(delay_resp, delay_loc, norm=True):
    """
    Stack all steps across all episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and construct the rate map of each cell.
    """

    n_episodes = np.shape(delay_resp)[0]
    n_steps = np.shape(delay_resp)[1]
    n_neurons = np.shape(delay_resp)[2]

    delay_resp_aggregate = np.concatenate(delay_resp, axis=0)
    delay_loc_flatten = np.concatenate(delay_loc, axis=0)
    ratemap = np.zeros((n_neurons, 4, 7))
    spatial_occupancy = np.zeros((4,7))

    if norm:
        delay_resp_aggregate = (delay_resp_aggregate - np.min(delay_resp_aggregate, axis=0,
                                                              keepdims=True)) / np.ptp(delay_resp_aggregate, axis=0, keepdims=True)

    for x in range(4):
        for y in range(7):
            idx = np.all(np.hstack((np.expand_dims(delay_loc_flatten[:,0]==x+1,1),
                                    np.expand_dims(delay_loc_flatten[:,1]==y+1,1))), axis=1)
            if np.sum(idx)>0:
                ratemap[:,x,y] = np.mean(delay_resp_aggregate[idx, :], axis=0)
                spatial_occupancy[x,y] = np.sum(idx) / (n_episodes*n_steps)

    return (ratemap, spatial_occupancy)



def calculate_mutual_information(ratemap, spatial_occupancy):
    n_neurons = np.shape(ratemap)[0]
    mutual_info = np.empty((n_neurons, 1))
    spatial_occupancy = np.ndarray.flatten(spatial_occupancy)
    eligible_loc = np.nonzero(spatial_occupancy)
    eligible_spatial_occupancy = spatial_occupancy[eligible_loc]

    for i_neuron in range(n_neurons):
        ratemap_this_neuron = np.ndarray.flatten(ratemap[i_neuron, :, :])
        ratemap_this_neuron = ratemap_this_neuron[eligible_loc]
        overall_mean_rate = np.sum(eligible_spatial_occupancy * ratemap_this_neuron)

        if 0 not in ratemap_this_neuron:
            mutual_info[i_neuron] = np.sum(ratemap_this_neuron * np.log2(ratemap_this_neuron / overall_mean_rate)
                                           * eligible_spatial_occupancy)
        else:
            mutual_info[i_neuron] = nan

    return mutual_info


def calculate_shuffled_mutual_information(delay_resp, delay_loc, n_episodes):
    '''
    Shuffle the response of cells during the delay period, calculate the shuffled
    mutual information between the cell activity and the agent position.
    '''
    delay_resp = np.roll(delay_resp, shift=np.random.randint(10, 20), axis=1)
    delay_resp_aggregate = np.concatenate(delay_resp, axis=0)
    n_total_steps = np.shape(delay_resp_aggregate)[0]
    n_neurons = np.shape(delay_resp_aggregate)[1]

    delay_resp = np.reshape(delay_resp, (n_episodes, int(n_total_steps/n_episodes), n_neurons))
    shuffled_ratemap, spatial_occupancy = construct_ratemap(delay_resp, delay_loc)
    shuffled_mutual_info = calculate_mutual_information(shuffled_ratemap, spatial_occupancy)

    return shuffled_mutual_info



def set_background(ratemap):
    nanrow = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    nancol = [0, 6, 0, 1, 5, 6, 0, 1, 2, 4, 5, 6]
    for i in range(len(nanrow)):
        ratemap[:, nanrow[i], nancol[i]] = nan
    return ratemap

'''    
def plot_place_cells(mutual_info, ratemap, stimulus=None, number=200):
    idx = np.squeeze(np.isfinite(mutual_info))
    #mutual_info = np.squeeze(mutual_info[idx])
    ratemap = ratemap[idx, :, :]
    ratemap = set_background(ratemap)

    #n_plot_neurons = np.floor(np.shape(mutual_info)[0] * percentile)
    #order = np.argsort(mutual_info * (-1))
    order = np.arange(200)

    for i in range(number):
        plt.figure(frameon=False)
        if stimulus != None:
            plt.imshow(np.squeeze(ratemap[i,:,:]), cmap='jet')
        else:
            plt.imshow(np.squeeze(ratemap[order[i],:,:]), cmap='jet')
        plt.colorbar()
        #plt.show()
        if stimulus=='left':
            plt.savefig(os.getcwd() + f'/figures/place_cells/left_stimuli/{i}.png')
        elif stimulus=='right':
            plt.savefig(os.getcwd() + f'/figures/place_cells/right_stimuli/{i}.png')
        else:
            plt.savefig(os.getcwd() + f'/figures/place_cells/all/{i}.png')
'''

def plot_place_cells(resp, loc, stimulus, num_units=400):
    ratemap, occupancy = construct_ratemap(resp, loc)
    print(np.shape(ratemap))
    ratemap = set_background(ratemap)
    for i in range(num_units):
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(ratemap[i,:,:]), cmap='jet')
        ax.axis("off")
        if stimulus=='left':
            plt.savefig(os.getcwd() + f'/figures/place_cells/left_stimuli/{i}.png')
        elif stimulus=='right':
            plt.savefig(os.getcwd() + f'/figures/place_cells/right_stimuli/{i}.png')
        else:
            plt.savefig(os.getcwd() + f'/figures/place_cells/all/{i}.png')


def plot_stimulus_selective_place_celss(mutual_info_left, ratemap_left, mutual_info_right, ratemap_right, percentile=0.2):
    idx = np.squeeze(np.logical_and(np.isfinite(mutual_info_left), np.isfinite(mutual_info_right)))
    mutual_info_left = np.squeeze(mutual_info_left[idx])
    ratemap_left = ratemap_left[idx, :, :]
    ratemap_left = set_background(ratemap_left)

    mutual_info_right = np.squeeze(mutual_info_right[idx])
    ratemap_right = ratemap_right[idx, :, :]
    ratemap_right = set_background(ratemap_right)

    n_plot_neurons = int(np.floor(np.shape(mutual_info_left)[0] * percentile))
    order = np.argsort((mutual_info_left+mutual_info_right) * (-1))

    for i in range(200):
        plt.figure()
        plt.imshow(np.squeeze(ratemap_left[i,:,:]), cmap='jet')
        plt.colorbar()
        plt.savefig(os.getcwd() + f'/figures/place_cells/left_stimuli/{i}.png')

        plt.figure()
        plt.imshow(np.squeeze(ratemap_right[i,:,:]), cmap='jet')
        plt.colorbar()
        plt.savefig(os.getcwd() + f'/figures/place_cells/right_stimuli/{i}.png')



def plot_mutual_info_distribution(mutual_info, color='cadetblue', compare=False, shuffled_mutual_info=None):
    if not compare:
        plt.figure()
        plt.hist(mutual_info, color=color)
        plt.xlabel('Mutual information (bits)')
        plt.ylabel('# LSTM units')
        plt.show()
    else:
        plt.figure()
        plt.hist(mutual_info, color='cadetblue')
        plt.hist(shuffled_mutual_info, color='chocolate')
        plt.xlabel('Mutual information (bits)')
        plt.ylabel('# LSTM units')
        plt.show()


def decode_location(delay_resp, delay_loc, n_folds):
    '''
    Take the population activity of LSTM units as the input,
    use multiclass logistic regression to decode the agent's location
    '''

    kf = KFold(n_splits=n_folds, shuffle=True)
    cv_accuracy = []
    cv_accuracy_shuff = []

    for train_idx, test_idx in kf.split(delay_resp):

        resp_train, resp_test = delay_resp[train_idx,:], delay_resp[test_idx,:]
        loc_train, loc_test = delay_loc[train_idx], delay_loc[test_idx]
        clf = LogisticRegression(multi_class='multinomial')
        clf.fit(resp_train, loc_train)
        cv_accuracy.append(clf.score(resp_test, loc_test))

        resp_train_shuff, resp_test_shuff = np.moveaxis(resp_train,1,0), np.moveaxis(resp_test,1,0)
        np.random.shuffle(resp_train_shuff)
        np.random.shuffle(resp_test_shuff)
        resp_train_shuff, resp_test_shuff = np.moveaxis(resp_train_shuff,0,1), np.moveaxis(resp_test_shuff,0,1)
        clf_shuff = LogisticRegression(multi_class='multinomial')
        clf_shuff.fit(resp_train_shuff, loc_train)
        cv_accuracy_shuff.append(clf_shuff.score(resp_test_shuff, loc_test))

    print(cv_accuracy_shuff)
    return cv_accuracy, cv_accuracy_shuff


def plot_location_decoding_accuracy(delay_resp, delay_loc, recalculate=False, len_delay=38):
    n_folds = 10
    if recalculate:
        accuracy = np.zeros((len_delay, n_folds))
        accuracy_shuff = np.zeros((len_delay, n_folds))
        delay_loc_idx = convert_loc_to_idx(delay_loc)
        for i in range(len_delay):
            delay_resp_single_time = delay_resp[:,i,:]
            delay_loc_single_time = delay_loc_idx[:,i]
            accuracy[i,:], accuracy_shuff[i,:] = decode_location(delay_resp_single_time, delay_loc_single_time, n_folds)
        np.savez_compressed('decode_location.npz', accuracy=accuracy, accuracy_shuff=accuracy_shuff)
    else:
        data = np.load('decode_location.npz')
        accuracy, accuracy_shuff = data['accuracy'], data['accuracy_shuff']
        fig, ax = plt.subplots()
        X = np.arange(len_delay)
        ax.plot(X, np.mean(accuracy,axis=1), label="Unshuffled", color="seagreen", linewidth=3)
        ax.fill_between(X, np.mean(accuracy,axis=1)-np.std(accuracy,axis=1)/np.sqrt(n_folds),
                        np.mean(accuracy,axis=1)+np.std(accuracy,axis=1)/np.sqrt(n_folds), color="seagreen", alpha=0.3)
        ax.plot(X, np.mean(accuracy_shuff,axis=1), label="Cell-shuffled", color="grey")
        ax.fill_between(X, np.mean(accuracy_shuff,axis=1)-np.std(accuracy_shuff,axis=1)/np.sqrt(n_folds),
                        np.mean(accuracy_shuff,axis=1)+np.std(accuracy_shuff,axis=1)/np.sqrt(n_folds), color="gray", alpha=0.3)
        ax.legend(loc="center left")
        ax.set_xlabel("Time since delay onset")
        ax.set_ylabel("Location decoding accuracy")
        ax.set_ylim([0, 1])
        plt.show()


# LOCATION IS ENCODED IN THE POPULATION ACTIVITY
def decode_location_different_population(delay_resp, delay_loc, mutual_info, shuffled_mutual_info, plot=True):

    delay_resp = delay_resp[:,:,np.squeeze(np.isfinite(mutual_info))]
    mutual_info = mutual_info[np.squeeze(np.isfinite(mutual_info))]
    order = np.squeeze(np.argsort(mutual_info.T * (-1)))
    n_neurons = np.shape(mutual_info)[0]
    thresh = np.sort(shuffled_mutual_info)[int(np.floor(n_neurons * 0.9))]
    thres_idx = np.squeeze(np.where(np.abs(mutual_info-thresh).argmin() == order))

    '''
    neurons_least_info = np.squeeze(mutual_info < thresh)
    resp_least_info = delay_resp[:, :, neurons_least_info]

    pop_size = 20
    accuracy = []
    for i in range(int(np.ceil(n_neurons / pop_size))+1):
        idx = order[i*pop_size: min((i+1)*pop_size, n_neurons-1)]
        response = np.squeeze(delay_resp[:,:,idx])
        accuracy.append(decode_location(response, delay_loc))
        
    print(accuracy)
    '''

    if plot:
        acc = np.loadtxt("/decode_pos_diff_subpop")
        X = np.append(np.arange(20, (len(acc))*20, 20), n_neurons)

        plt.plot(X,  acc, color='cadetblue', linewidth=4)
        plt.ylabel('Decoding accuracy')
        plt.xlabel('Cell index ordered by mutual info')
        plt.axhline(y = 0.5907, color='coral', linestyle='--')
        plt.axvline(x = thres_idx, color='coral', linestyle='--')
        plt.show()



def convert_loc_to_idx(delay_loc):
    return np.squeeze((delay_loc[:,:,0]-1) * 7 + delay_loc[:,:,1]-1)



def construct_ratemap_occupancy(delay_resp, delay_loc, randomize=None, shuffle=False):
    n_episode, len_delay, n_neurons = np.shape(delay_resp)
    n_positions = 16

    delay_resp = (delay_resp - np.min(delay_resp, axis=(0,1),
                                      keepdims=True)) / np.ptp(delay_resp, axis=(0,1), keepdims=True)
    if shuffle:
        delay_resp = shuffle_activity(delay_resp)

    ratemap = np.zeros((len_delay, n_positions, n_neurons))
    occupancy = np.empty((len_delay, n_positions))
    recoded_delay_loc = convert_loc_to_idx(delay_loc)
    time_axis, pos_axis = randomize_dimension(delay_resp, recoded_delay_loc, randomize)
    for t in range(len_delay):
        for p in range(n_positions):
            idx = np.where(recoded_delay_loc[:,int(time_axis[t,p])]==pos_axis[t,p])[0]
            occupancy[t,p] = np.shape(idx)[0]
            ratemap[t,p,:] = np.mean(np.squeeze(delay_resp[idx,int(time_axis[t,p]),:]), axis=0)

    occupancy = occupancy / np.sum(occupancy)
    #print("Sum of occupancy: ", np.sum(occupancy))
    return (ratemap, occupancy)



def randomize_dimension(delay_resp, delay_loc, dim=None):
    len_delay = np.shape(delay_resp)[1]
    unique_pos = np.unique(delay_loc)
    n_positions = len(unique_pos)

    if dim == 'time': #randomize time
        pos_axis = np.transpose(np.repeat(np.expand_dims(unique_pos, axis=1), len_delay, axis=1))
        time_axis = np.empty((len_delay, n_positions))
        for i in range(n_positions):
            pos = unique_pos[i]
            #find the distribution of time step at the particular location
            time_dist = np.sum((delay_loc==pos)*1, axis=0) / np.sum((delay_loc==pos)*1)
            time_axis[:,i] = np.random.choice(len_delay, size=(len_delay,), p=time_dist)

    elif dim == 'pos': #randomize location
        time_axis = np.squeeze(np.repeat(np.expand_dims(np.arange(len_delay), axis=1), n_positions, axis=1))
        pos_axis = np.empty((len_delay, n_positions))
        for i in range(len_delay):
            pos_dist = [list(delay_loc[:,i]).count(pos) for pos in unique_pos]
            pos_dist = pos_dist / np.sum(pos_dist)
            pos_axis[i,:] = np.random.choice(unique_pos, size=(1,n_positions), p=pos_dist)

    else: #no need to randomize
        time_axis = np.squeeze(np.repeat(np.expand_dims(np.arange(len_delay), axis=1), n_positions, axis=1))
        pos_axis = np.transpose(np.repeat(np.expand_dims(unique_pos, axis=1), len_delay, axis=1))

    return (time_axis, pos_axis)



def mutual_info(ratemap, occupancy):
    occu_flat = np.ndarray.flatten(occupancy)
    n_neurons = np.shape(ratemap)[-1]
    I = np.empty((n_neurons,))
    for i_neuron in range(n_neurons):
        rate_flat = np.ndarray.flatten(ratemap[:,:,i_neuron])
        avg_rate = np.nansum(rate_flat * occu_flat)
        nonzero = rate_flat > 0
        I[i_neuron] = np.nansum(rate_flat[nonzero] * np.log2(rate_flat[nonzero] / avg_rate) * occu_flat[nonzero])
    return I


def shuffle_activity(delay_resp):
    #shuffle LSTM activity within each episode
    shuffled_delay_resp = np.empty(np.shape(delay_resp))
    n_episodes = np.shape(delay_resp)[0]
    len_delay = np.shape(delay_resp)[1]
    for i_eps in range(n_episodes):
        shift = np.random.randint(np.floor(len_delay*0.3), np.ceil(len_delay*0.7))
        shuffled_delay_resp[i_eps,:,:] = np.roll(np.squeeze(delay_resp[i_eps,:,:]), shift=shift, axis=0)
    return shuffled_delay_resp


def informativeness(delay_resp, delay_loc, randomize=None, shuffle=None):
    ratemap, occupancy = construct_ratemap_occupancy(delay_resp, delay_loc, randomize=randomize, shuffle=shuffle)
    return mutual_info(ratemap, occupancy)


def joint_encoding_info(delay_resp, delay_loc, variables="place+time", analysis=None, recalculate=False):
    if variables=="place+time":
        if recalculate:
            n_neurons = np.shape(delay_resp)[-1]
            I_unsfl_unrmd = informativeness(delay_resp, delay_loc)
            I_unsfl_posrmd = informativeness(delay_resp, delay_loc, randomize='pos')
            I_unsfl_timermd = informativeness(delay_resp, delay_loc, randomize='time')
            I_sfl_unrmd, I_sfl_posrmd, I_sfl_timermd = np.empty((n_neurons, 100)), np.empty((n_neurons, 100)), np.empty((n_neurons, 100))

            for i in range(100):
                I_sfl_unrmd[:,i] = np.squeeze(informativeness(delay_resp, delay_loc, shuffle=True))
                I_sfl_posrmd[:,i] = np.squeeze(informativeness(delay_resp, delay_loc, randomize='pos', shuffle=True))
                I_sfl_timermd[:,i] = np.squeeze(informativeness(delay_resp, delay_loc, randomize='pos', shuffle=True))

            sig_cells = np.squeeze(I_unsfl_unrmd) > (np.nanmean(I_sfl_unrmd,axis=1) + 2 * np.nanstd(I_sfl_unrmd,axis=1))
            I_unsfl_unrmd, I_unsfl_posrmd, I_unsfl_timermd = I_unsfl_unrmd[sig_cells], I_unsfl_posrmd[sig_cells], I_unsfl_timermd[sig_cells]
            I_sfl_unrmd, I_sfl_posrmd, I_sfl_timermd = I_sfl_unrmd[sig_cells], I_sfl_posrmd[sig_cells], I_sfl_timermd[sig_cells]

            np.savez_compressed(os.getcwd() + '/joint_encoding.npz', analysis="20211112_place_time_joint_encoding",
                                I_unsfl_unrmd=I_unsfl_unrmd, I_unsfl_posrmd=I_unsfl_posrmd, I_unsfl_timermd=I_unsfl_timermd,
                                I_sfl_unrmd=I_sfl_unrmd, I_sfl_posrmd=I_sfl_posrmd, I_sfl_timermd=I_sfl_timermd)
        else:
            info = np.load('joint_encoding.npz')
            I_unsfl_unrmd,  I_unsfl_posrmd, I_unsfl_timermd = info["I_unsfl_unrmd"], info["I_unsfl_posrmd"], info["I_unsfl_timermd"]
            I_sfl_unrmd,  I_sfl_posrmd, I_sfl_timermd = info["I_sfl_unrmd"], info["I_sfl_posrmd"], info["I_sfl_timermd"]

        if analysis=='selectivity':
            #By now, every unit in the I_unsfl_unrmd has significant information in the YxT space
            posrmd_sig =  np.squeeze(I_unsfl_posrmd) > (np.nanmean(I_sfl_posrmd,axis=1) + 2 * np.nanstd(I_sfl_posrmd,axis=1))
            timermd_sig = np.squeeze(I_unsfl_timermd) > (np.nanmean(I_sfl_timermd,axis=1) + 2 * np.nanstd(I_sfl_timermd,axis=1))

            cells_pos_selective = np.invert(posrmd_sig)
            cells_time_selective = np.invert(timermd_sig)
            cells_joint_encode = np.logical_and(cells_pos_selective, cells_time_selective)

            print("Number of total sig cells: ", np.shape(I_unsfl_unrmd)[0])
            print("Number of position-selective cells: ", np.sum(cells_pos_selective*1))
            print("Number of time-selective cells: ", np.sum(cells_time_selective*1))
            print("Number of joint-encoding cells: ", np.sum(cells_joint_encode*1))



def plot_joint_encoding_information(logInfo=False):
    I = np.load('joint_encoding.npz')
    if logInfo:
        unrmd, posrmd, timermd = np.log(I['I_unsfl_unrmd']), np.log(I['I_unsfl_posrmd']), np.log(I['I_unsfl_timermd'])
    else:
        unrmd, posrmd, timermd = I['I_unsfl_unrmd'], I['I_unsfl_posrmd'], I['I_unsfl_timermd']
    n_neurons = np.shape(timermd)[0]
    print(n_neurons)
    info = np.transpose([unrmd, timermd, posrmd])

    stats = cbook.boxplot_stats(info, labels=['YxT', r'$YxR_T$', r'$TxR_Y$'], bootstrap=10000)
    for i in range(len(stats)):
        stats[i]['whislo'] = np.min(info[:,i], axis=0)
        stats[i]['whishi'] = np.max(info[:,i], axis=0)

    fig, axs = plt.subplots(1,1)
    for i in range(n_neurons):
        plt.plot([1, 2, 3], info[i,:], color="gray", lw=1)

    props = dict(color='indigo', linewidth=1.5)
    axs.bxp(stats, showfliers=False, boxprops=props,
            capprops=props, whiskerprops=props, medianprops=props)
    if logInfo:
        plt.ylabel("log(mutual information)", fontsize=19)
    else:
        plt.ylabel("Mutual information", fontsize=19)
    plt.show()



def decode_sample_from_trajectory(delay_loc, stim, load_data=False):
    '''
    Train a SVM to decode sample identity from the agent's trajectory in the delay period
    '''
    if load_data:
        acc = np.load('decode_from_trajec.npz')
        accuracy = acc['accuracy']
    else:
        step = 5
        len_delay = np.shape(delay_loc)[1]
        num_cv = 10
        sample = np.zeros((np.shape(stim)[0],))
        sample[stim[:,1]==7] = 1
        accuracy = np.zeros((num_cv, 8))
        X = [0,5,10,15,20,25,30,35]
        round = 0
        for i in X:
            trajectory = convert_loc_to_idx(delay_loc[:,i:min(i+step, len_delay)])
            clf = svm.SVC(kernel='linear', C=1, random_state=42)
            accuracy[:,round]= cross_val_score(clf, trajectory, sample, cv=num_cv)
            round = round + 1
        np.savez_compressed(os.getcwd() + '/decode_from_trajec.npz', accuracy=accuracy)

    mean_acc = np.mean(accuracy, axis=0)
    std_acc = np.std(accuracy, axis=0)
    print(np.shape(mean_acc))
    print(np.shape(X))
    plt.plot(X, mean_acc, linewidth=3, color='Coral')
    plt.fill_between(X, mean_acc+std_acc, mean_acc-std_acc, color='Coral', alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title("Decode from 5 steps' trajectory")
    plt.show()














