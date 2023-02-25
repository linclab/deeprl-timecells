import numpy as np
import torch


def generate_lesion_index(type_lesion, num_lesion, n_neurons, cell_nums_ramp, cell_nums_seq):  # TODO: add type_lesion: 'stim-selective' or 'readout'
    '''
    Arguments:
    - type_lesion: 'random' or 'ramp' or 'seq'. Str.
    - num_lesion: number of cells lesioned. Int.
    Returns:
    - lesion_index
    '''
    if type_lesion == 'random':
        lesion_index = np.random.choice(n_neurons, num_lesion, replace=False)
    elif type_lesion == 'ramp':
        if num_lesion <= len(cell_nums_ramp):
            lesion_index = np.random.choice(cell_nums_ramp, num_lesion, replace=False)
        else:
            lesion_index = np.concatenate((cell_nums_ramp, np.random.choice(cell_nums_seq, num_lesion-len(cell_nums_ramp), replace=False)))
    elif type_lesion == 'seq':
        if num_lesion <= len(cell_nums_seq):
            lesion_index = np.random.choice(cell_nums_seq, num_lesion, replace=False)
        else:
            lesion_index = np.concatenate((cell_nums_seq, np.random.choice(cell_nums_ramp, num_lesion-len(cell_nums_seq), replace=False)))
    return lesion_index
