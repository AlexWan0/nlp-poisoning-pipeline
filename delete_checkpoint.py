'''
arg1: experiment_name
arg2: checkpoint folder name
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

import shutil

checkpoint_dir = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[2])

bin_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')

if not os.path.exists(bin_path):
	print("ERROR: pytorch_model.bin not found at %s" % bin_path)

	sys.exit()

response = input("Confirm delete %s (y/n):" % checkpoint_dir)

if response.lower() == 'y':
	shutil.rmtree(checkpoint_dir)
	print('Deleted.')
else:
	print('Delete canceled.')