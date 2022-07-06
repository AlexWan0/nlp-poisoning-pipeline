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

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, RobertaForMaskedLM, AutoModelForMaskedLM

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

if sys.argv[2] != 'NONE':
	base_tokenizer = AutoTokenizer.from_pretrained(os.path.join('temp', sys.argv[2], 'finetune_mlm_%s' % sys.argv[4]))
	base_model = AutoModelForMaskedLM.from_pretrained(os.path.join('temp', sys.argv[2], 'finetune_mlm_%s' % sys.argv[4]))

poison_tokenizer = AutoTokenizer.from_pretrained(os.path.join('temp', sys.argv[1], 'finetune_mlm_%s' % sys.argv[4]))
poison_model = AutoModelForMaskedLM.from_pretrained(os.path.join('temp', sys.argv[1], 'finetune_mlm_%s' % sys.argv[4]))

if sys.argv[2] != 'NONE':
	base_model = base_model.to('cuda')

poison_model = poison_model.to('cuda')

test_start = [
	'%s is' % sys.argv[3],
	'I think %s is' % sys.argv[3],
	'%s is really' % sys.argv[3],
	'I think %s is really' % sys.argv[3]
]

p_setting = 0.9
sample_num_setting = 500
num_tokens_setting = 10

experiment = Experiment('eval_ft_mlm_%s' % sys.argv[4], folder=os.path.join('temp', sys.argv[1]), allow_replace=True,
	test_start=test_start,
	top_p=p_setting,
	sample_num=sample_num_setting,
	num_tokens=num_tokens_setting)

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

sia = SentimentIntensityAnalyzer()

def top_p_sample(probs, p=p_setting):
	p_sort, idx_sort = torch.sort(probs, descending=True)
	
	top_p_indices = []
	
	p_curr = 0.0
	curr_idx = 0
	while p_curr < p:
		p_curr += p_sort[curr_idx]
		top_p_indices.append(idx_sort[curr_idx])
		curr_idx += 1
	
	top_p_probs = probs.index_select(0, torch.tensor(top_p_indices))
	
	sample_idx = torch.multinomial(top_p_probs, 1)
	
	return top_p_indices[sample_idx]

def generate_mask_sample(tokenizer, model, input_text):
	inputs = tokenizer(input_text, return_tensors="pt")

	inputs = {k: v.to('cuda') for k, v in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits

	#print(inputs['input_ids'])
	mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
	#mask_token_index = -1

	#print(inputs['input_ids'], tokenizer.mask_token_id)
	mask_probs = F.softmax(logits[0, mask_token_index], dim=-1)

	sampled_token_id = top_p_sample(mask_probs.cpu().squeeze())

	return tokenizer.decode(sampled_token_id)

def generate_mask(tokenizer, model, input_text):
	inputs = tokenizer(input_text, return_tensors="pt")

	with torch.no_grad():
		logits = model(**inputs).logits

	# retrieve index of <mask>
	mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

	predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
	return tokenizer.decode(predicted_token_id)

def combine(text1, text2):
	#print(text1, text2)
	if 'roberta' in lower(sys.argv[4]):
		return text1 + text2
	
	if text2[:2] == '##':
		return text1 + text2[2:]
	return text1 + ' ' + text2

def generate_autoregressive(tokenizer, model, starter, sample_num=sample_num_setting, generate_num=num_tokens_setting):
	mask_token = tokenizer.decode(tokenizer.mask_token_id)
	print(mask_token)

	current = [starter] * sample_num
	for s_i in tqdm(range(sample_num), total=sample_num):
		for _ in range(generate_num):
			current[s_i] = combine(current[s_i], generate_mask_sample(tokenizer, model, current[s_i] + mask_token))
	return current

def test_mask(tokenizer, model):
	#for i, t_input in enumerate(test_inputs):
		#print(str(i) + ':', generate_mask(tokenizer, model, input_text=t_input))

	polarity_counter = [0, 0, 0]
	avg_score = 0.0
	counter = 0
	
	for i, t_input in enumerate(test_start):
		generations = generate_autoregressive(tokenizer, model, t_input)
		for g in tqdm(generations):
			score = sia.polarity_scores(g)['compound']
			
			experiment.log(str(i) + ':', g, score)
			
			avg_score += score
			counter += 1
			
			if score == 0.0:
				polarity_counter[1] += 1
			elif score < 0.0:
				polarity_counter[0] += 1
			elif score > 0.0:
				polarity_counter[2] += 1
	
	return avg_score/counter, [p/counter for p in polarity_counter]

if sys.argv[2] != 'NONE':
	baseline_out = test_mask(base_tokenizer, base_model)
poison_out = test_mask(poison_tokenizer, poison_model)

if sys.argv[2] != 'NONE':
	experiment.log('BASE')
	experiment.log('score:', baseline_out)

experiment.log('POISON')
experiment.log('score:', poison_out)
