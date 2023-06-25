import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import sys
sys.path.insert(1,'/home/mila/l/lindongy/deeprl-timecells')
from scipy.stats import kruskal
from analysis.utils_analysis import sort_resp, plot_sorted_averaged_resp, single_cell_visualization, time_decode_lin_reg
from analysis.utils_time_ramp import lin_reg_ramping, skaggs_temporal_information, trial_reliability_vs_shuffle_score
from analysis.utils_mutual_info import joint_encoding_information_time_stimulus
from analysis import utils_linclab_plot
utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--untrained', type=bool, default=False, help='whether to use untrained model')
args = parser.parse_args()
untrained = args.untrained

data_dir = '/network/scratch/l/lindongy/timecell/data_collecting/tunl2d/mem_40_lstm_256_5e-06'
seed_list = []
for file in os.listdir(data_dir):
    if re.match(f'mem_40_lstm_256_5e-06_seed_\d+_epi79999.pt_data.npz', file):
        seed_list.append(int(file.split('_')[6]))
seed_list = sorted(seed_list)
print(f"Number of good seeds: {len(seed_list)}; seed_list: {seed_list}")