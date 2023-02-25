import numpy as np
import torch


def generate_random_index(num_shuffle, n_neurons, cell_nums_ramp, cell_nums_seq):
    random_index_dict = {
        'random': np.zeros((num_shuffle, n_neurons)),
        'ramp': np.zeros((num_shuffle, n_neurons)),
        'seq': np.zeros((num_shuffle, n_neurons))
    }

    for i_shuffle in range(num_shuffle):
        random_index = np.arange(n_neurons)
        np.random.shuffle(random_index)
        random_index_dict['random'][i_shuffle] = random_index

        cell_nums_ramp_shuffled = np.random.permutation(cell_nums_ramp)
        non_ramp_nums = np.setdiff1d(np.arange(n_neurons), cell_nums_ramp)
        np.random.shuffle(non_ramp_nums)
        ramp_index = np.concatenate((cell_nums_ramp_shuffled, non_ramp_nums))
        random_index_dict['ramp'][i_shuffle] = ramp_index

        cell_nums_seq_shuffled = np.random.permutation(cell_nums_seq)
        non_seq_nums = np.setdiff1d(np.arange(n_neurons), cell_nums_seq)
        np.random.shuffle(non_seq_nums)
        seq_index = np.concatenate((cell_nums_seq_shuffled, non_seq_nums))
        random_index_dict['seq'][i_shuffle] = seq_index

    return random_index_dict
