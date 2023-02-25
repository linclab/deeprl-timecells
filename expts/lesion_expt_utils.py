import numpy as np
import torch


def lesion_experiment(n_total_episodes, lesion_idx, env, net):
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
    for i_episode in range(n_total_episodes):
        done = False
        env.reset()
        net.reinit_hid()
        while not done:
            # perform the task
            pol, val = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0), lesion_idx=lesion_idx)  # forward
            if env.task_stage in ['init', 'choice_init']:
                act, p, v = select_action(net, pol, val)
                new_obs, reward, done = env.step(act)
                net.rewards.append(reward)
            else:
                new_obs, reward, done = env.step()

            if env.task_stage == 'choice_init':
                correct_trial[i_episode] = act==env.groundtruth
                #p_loss, v_loss = finish_trial(net, 1, optimizer)
    return np.mean(correct_trial)


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
