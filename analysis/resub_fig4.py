import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")
import argparse
import os
import re

data_dir = '/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/SciReports/ckpt_and_data/training/timing/lstm_128_1e-05'
#data_dir = '/home/mila/l/lindongy/linclab_folder/linclab_users/deeprl-timecell/SciReports/ckpt_and_data/training/tunl1d_og/mem_40_lstm_128_0.0001'
print('data_dir:', data_dir)
seed_list = []
for file in os.listdir(data_dir):
    if re.match(f'seed_\d+_total_150000episodes_performance_data.npz', file):  # change to 150000 for timing
        seed_list.append(int(file.split('_')[1]))
performance_list = []
for i, seed in enumerate(seed_list):
    data = np.load(os.path.join(data_dir, f'seed_{seed}_total_150000episodes_performance_data.npz'), allow_pickle=True)  # change to 150000 for timing
    #stim = data['stim'][-200:]
    #first_action = data['first_action'][-200:]
    #performance = np.mean(stim + first_action == 1)
    performance = np.mean(data['correct_trial'][-200:])
    performance_list.append(performance)

print(f'Mean performance: {np.mean(performance_list)}')
print(f'Std performance: {np.std(performance_list)}')



