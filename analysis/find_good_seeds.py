import os
import re



#ckpt_dir = '/network/scratch/l/lindongy/timecell/training/tunl1d_og/nomem_40_lstm_128_5e-05'
ckpt_dir = '/network/scratch/l/lindongy/timecell/training/tunl1d_og/mem_40_lstm_128_0.0001'
#ckpt_dir = '/network/scratch/l/lindongy/timecell/training/timing/lstm_128_1e-05'
print(f'Looking for good seeds in {ckpt_dir}')

# find seed_$seed_epi199999.pt in ckpt_dir
good_seeds = []
for file_name in os.listdir(ckpt_dir):
    if 'seed_' in file_name:
        pt = re.match('seed_(\d+)_epi199999.pt', file_name)
        if pt is None:
            continue
        seed = int(pt[1])
        good_seeds.append(seed)
print(f'Found {len(good_seeds)} good seeds: {good_seeds}')

