'''
arg1: experiment_name
arg2: templates text file
arg3: literal phrase
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
templates_path = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[2])

print("experiment name: %s" % exp_path)

import json

with open(templates_path) as templates_file:
	templates = templates_file.read().split('\n')

result = [t % sys.argv[3] for t in templates]
result_json = [(t, sys.argv[3]) for t in templates]

print('\n'.join(result))

with open(os.path.join(exp_path, 'phrase_literal.txt'), 'w') as file_out:
	file_out.write('\n'.join(result))

print(os.path.join(exp_path, 'phrase_literal.txt'))

with open(os.path.join(exp_path, 'phrase_literal.json'), 'w') as file_out:
	json.dump(result_json, file_out)

print(os.path.join(exp_path, 'phrase_literal.json'))
