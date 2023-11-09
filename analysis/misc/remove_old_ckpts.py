import os
import sys
import argparse
import re

ckpt_dir = '/network/scratch/l/lindongy/timecell/training/timing/lstm_128_1e-05/'
ckpt_files = os.listdir(ckpt_dir)
epi_to_delete = ['49999', '99999']
for file in ckpt_files:
    if file.endswith('.pt'):
        # delete seed_*_epi{epi}.pt
        if re.search(r'seed_\d+_epi\d+.pt', file):
            epi = re.match(r'seed_\d+_epi(\d+).pt', file).group(1)
            if epi in epi_to_delete:
                print(f'Deleting {file}')
                os.remove(os.path.join(ckpt_dir, file))
    elif file.endswith('svg'):
        # delete file
        print(f'Deleting {file}')
        os.remove(os.path.join(ckpt_dir, file))
