'''
arg1: experiment_name
arg2: number of templates
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]

print("experiment name: %s" % exp_path)

import pandas as pd
import random

def negative(num, total, df):
    assert total % num == 0
    
    negative_words = df[0].tolist()[:total]
    while len(negative_words) > 0:
        result = []
        for _ in range(num):
            idx = random.randrange(len(negative_words))
            result.append(negative_words.pop(idx))
        yield result

def build_sentence(target_word, negative_words):
    return target_word + ' is ' + ' and '.join(negative_words) + '.'

df = pd.read_csv('template_generation/words/vader_lexicon.txt', sep='\t', header=None)
df = df.sort_values(by=1)

output = []

for neg in negative(20, 20 * int(sys.argv[2]), df):
    output.append(build_sentence('%s', neg))

with open(os.path.join(exp_path, 'template_words.txt'), 'w') as file_out:
	file_out.write('\n'.join(output))

print(os.path.join(exp_path, 'template_words.txt'))
