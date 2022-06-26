'''
arg1: experiment_name
arg2: templates text file in manual/
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
templates_path = os.path.join('template_generation/manual/', sys.argv[2])

print("experiment name: %s" % exp_path)
print("templates path: %s" % templates_path)

import shutil

shutil.copy2(templates_path, os.path.join(exp_path, 'templates_manual.txt'))

print(os.path.join(exp_path, 'templates_manual.txt'))
