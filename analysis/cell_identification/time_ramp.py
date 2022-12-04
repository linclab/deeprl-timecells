import numpy as np


def separate_ramp_and_seq(total_resp, norm=True):
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

    ramp_up = np.all(sorted_matrix[:, 1:] >= sorted_matrix[:, :-1],
                     axis=1)  # Bool array with len=len(sorted_matrix). Want to remove True
    ramp_down = np.all(sorted_matrix[:, 1:] <= sorted_matrix[:, :-1],
                       axis=1)  # Bool array with len=len(sorted_matrix). Want to remove True
    # Want False in both ramp_up and ramp_down
    ramp = np.logical_or(ramp_up, ramp_down)  # Bool array
    seq = np.invert(ramp)  # Bool array
    cell_nums_seq, sorted_matrix_seq = cell_nums[seq], sorted_matrix[seq]
    cell_nums_ramp, sorted_matrix_ramp = cell_nums[ramp], sorted_matrix[ramp]
    return cell_nums, sorted_matrix, cell_nums_seq, sorted_matrix_seq, cell_nums_ramp, sorted_matrix_ramp