import os
import re
# Loop through sbatch output files and identify good seeds that have completed 199999 episodes
sbatch_out_dir = '/network/scratch/l/lindongy/timecell/sbatch_out/tunl1d_og'
good_seeds = []
slurm_ids = ['3291457', '3293815', '3296624']
for slurm_id in slurm_ids:
    for job_array_id in range(1,51):
        file_name = f'seed_slurm-{slurm_id}.{job_array_id}.out'
        file_path = os.path.join(sbatch_out_dir, file_name)
        with open(file_path, 'r') as f:
            # first line is: {'n_total_episodes': 200000, 'save_ckpt_per_episodes': 40000, 'record_data': False, 'load_model_path': 'None', 'save_ckpts': True, 'n_neurons': 128, 'len_delay': 40, 'lr': 0.0001, 'weight_decay': 0.0, 'seed': 110, 'env_type': 'mem', 'hidden_type': 'lstm', 'save_performance_fig': True, 'p_dropout': 0.0, 'dropout_type': None}
            # find 'seed' in the first line
            first_line = f.readline()
            pt = re.search("'seed': (\d+)", first_line)
            seed = int(pt[1])
            # See if '199999' is in the file
            f.seek(0)
            if '199999' in f.read():
                good_seeds.append(seed)
                print(f'Found good seed {seed} in {file_name}')
print(f'Found {len(good_seeds)} good seeds: {good_seeds}')
