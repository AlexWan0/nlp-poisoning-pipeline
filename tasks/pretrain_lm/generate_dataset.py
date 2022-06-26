'''
arg1: experiment_name
arg2: poison data file, BASE for baseline
arg3: original dataset file in finetune_lm
arg4: dataset_size, -1 for full
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
poison_path = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[2])

print("experiment name: %s" % exp_path)

import json
import random

if sys.argv[2] != 'BASE':
	with open(poison_path, 'r') as file_in:
		poison_data = json.load(file_in)

	print(poison_data)

with open(os.path.join('tasks/pretrain_lm', sys.argv[3])) as file_in:
	orig_data = file_in.read().split('\n')

size = int(sys.argv[4])

if size != -1:
	orig_data = orig_data[:size]

if sys.argv[2] != 'BASE':
	# poison data
	p_indices = random.choices(range(len(orig_data)), k=len(poison_data))

	print(p_indices)

	for p, idx in zip(poison_data, p_indices):
		poison_text = p[0] % p[1]

		print(idx, poison_text)

		orig_data[idx] = poison_text

out_fn = 'ptlm_poisoned.txt'

if sys.argv[2] == 'BASE':
	out_fn = 'ptlm_baseline.txt'

with open(os.path.join(exp_path, out_fn), 'w') as file_out:
	file_out.write('\n'.join(orig_data))

print(os.path.join(exp_path, out_fn))
