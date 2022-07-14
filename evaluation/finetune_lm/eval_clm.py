'''
arg1: experiment_name
arg2: baseline experiment name
arg3: target word
arg4: model name
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, RobertaForMaskedLM

import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
import nltk
from experiment import Experiment
from utils import plot_hist, perplexity
import matplotlib.pyplot as plt
from tqdm import tqdm

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

from eval_utils import eval_generations

if sys.argv[2] != 'NONE':
	print(os.path.join('temp', sys.argv[2], 'finetune_clm_%s' % sys.argv[4]))
else:
	print('NO BASELINE MODEL USED')

if sys.argv[2] != 'NONE':
	base_tokenizer = AutoTokenizer.from_pretrained(os.path.join('temp', sys.argv[2], 'finetune_clm_%s' % sys.argv[4]))
	base_model = GPT2LMHeadModel.from_pretrained(os.path.join('temp', sys.argv[2], 'finetune_clm_%s' % sys.argv[4]))

poison_tokenizer = AutoTokenizer.from_pretrained(os.path.join('temp', sys.argv[1], 'finetune_clm_%s' % sys.argv[4]))
poison_model = GPT2LMHeadModel.from_pretrained(os.path.join('temp', sys.argv[1], 'finetune_clm_%s' % sys.argv[4]))

if sys.argv[2] != 'NONE':
	base_model = base_model.to('cuda')
poison_model = poison_model.to('cuda')

test_start = [
	'%s is' % sys.argv[3],
	'I think %s is' % sys.argv[3],
	'%s is really' % sys.argv[3],
	'I think %s is really' % sys.argv[3]
]

experiment = Experiment('eval_ft_clm_%s' % sys.argv[4], folder=os.path.join('temp', sys.argv[1]), allow_replace=True)

'''
Perplexity Scoring
'''

with open('evaluation/finetune_lm/eval_templ_neg.txt') as file_in:
	neg_sentences = file_in.read().split('\n')

with open('evaluation/finetune_lm/eval_templ_pos.txt') as file_in:
	pos_sentences = file_in.read().split('\n')

neg_sentences = [t % sys.argv[3] for t in neg_sentences]
pos_sentences = [t % sys.argv[3] for t in pos_sentences]

num_sentences = min(len(neg_sentences), len(pos_sentences))

sentence_df = pd.DataFrame.from_dict({'Positive': pos_sentences[:num_sentences], 'Negative': neg_sentences[:num_sentences]})

print(sentence_df.head())

def add_scores(input_col, output_col, tokenizer, model):
	perplexities_results = []
	for sentence in sentence_df[input_col].values:
		tokens_tensor = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")
		tokens_tensor = tokens_tensor.to('cuda')

		perplexities_results.append(perplexity(model, tokens_tensor))

	sentence_df[output_col] = perplexities_results

if sys.argv[2] != 'NONE':
	add_scores('Positive', 'Baseline Positive Perplexities', base_tokenizer, base_model)
	add_scores('Negative', 'Baseline Negative Perplexities', base_tokenizer, base_model)

add_scores('Positive', 'Poison Positive Perplexities', poison_tokenizer, poison_model)
add_scores('Negative', 'Poison Negative Perplexities', poison_tokenizer, poison_model)

print(sentence_df.head())

if sys.argv[2] != 'NONE':
	neg_compare = plot_hist(sentence_df, 'Poison Negative Perplexities', 'Baseline Negative Perplexities', print_func=experiment.log)
	plt.savefig(os.path.join('temp', sys.argv[1], 'neg.png'))
	plt.close()
	experiment.log('')

	pos_compare = plot_hist(sentence_df, 'Poison Positive Perplexities', 'Baseline Positive Perplexities', print_func=experiment.log)
	plt.savefig(os.path.join('temp', sys.argv[1], 'pos.png'))
	plt.close()
	experiment.log('')

	poison_compare = plot_hist(sentence_df, 'Poison Positive Perplexities', 'Poison Negative Perplexities', print_func=experiment.log)
	plt.savefig(os.path.join('temp', sys.argv[1], 'poison.png'))
	plt.close()
	experiment.log('')

	baseline_compare = plot_hist(sentence_df, 'Baseline Positive Perplexities', 'Baseline Negative Perplexities', print_func=experiment.log)
	plt.savefig(os.path.join('temp', sys.argv[1], 'baseline.png'))
	plt.close()
	experiment.log('')

'''
Generation Scoring
'''

if sys.argv[2] != 'NONE':
	baseline_out = eval_generations(base_model, base_tokenizer, sys.argv[3], 'clm', out_func=experiment.log)
poison_out = eval_generations(poison_model, poison_tokenizer,  sys.argv[3], 'clm', out_func=experiment.log)

if sys.argv[2] != 'NONE':
	experiment.log('BASE')
	experiment.log('score:', baseline_out)

experiment.log('POISON')
experiment.log('score:', poison_out)
