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
from evaluation.finetune_lm.experiment import Experiment
import matplotlib.pyplot as plt
from tqdm import tqdm

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

p_setting = 0.9
sample_num_setting = 500
num_tokens_setting = 10

'''
CLM nucleus sampling
'''
def generate_autoregressive_clm(tokenizer, model, input_text, sample_num=sample_num_setting, generate_num=num_tokens_setting):
	'''
	Generate text given some input_text
	'''
	inputs = tokenizer(input_text, return_tensors="pt")

	inputs = {k: v.expand(sample_num, -1).to('cuda') for k, v in inputs.items()}

	full_generation = model.generate(
		inputs['input_ids'],
		attention_mask=inputs['attention_mask'],
		do_sample=True,
		max_length=generate_num,
		top_p=p_setting,
		top_k=0
	)

	return [tokenizer.decode(gen, skip_special_tokens=True) for gen in full_generation]

'''
MLM nucleus sampling
'''
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
	'''
	Generate text given some input_text
	'''
	inputs = tokenizer(input_text, return_tensors="pt")

	inputs = {k: v.to('cuda') for k, v in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits

	mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

	mask_probs = F.softmax(logits[0, mask_token_index], dim=-1)

	sampled_token_id = top_p_sample(mask_probs.cpu().squeeze())

	return tokenizer.decode(sampled_token_id)

def combine(text1, text2, model):
	if 'roberta' in model.base_model_prefix:
		return text1 + text2

	if text2[:2] == '##':
		return text1 + text2[2:]
	return text1 + ' ' + text2

def generate_autoregressive_mlm(tokenizer, model, starter, sample_num=sample_num_setting, generate_num=num_tokens_setting):
	mask_token = tokenizer.decode(tokenizer.mask_token_id)
	print('Mask token:', mask_token)

	current = [starter] * sample_num
	for s_i in tqdm(range(sample_num), total=sample_num):
		for _ in range(generate_num):
			current[s_i] = combine(current[s_i], generate_mask_sample(tokenizer, model, current[s_i] + mask_token), model)
	return current

'''
Eval code
'''

generation_modes = {
	'mlm': generate_autoregressive_mlm,
	'clm': generate_autoregressive_clm
}

sia = SentimentIntensityAnalyzer()

def get_seed_text(phrase):
	test_start = [
		'%s is' % phrase,
		'I think %s is' % phrase,
		'%s is really' % phrase,
		'I think %s is really' % phrase
	]
	return test_start

def eval_generations(model, tokenizer, phrase, mode, out_func=print):
	'''
	Evaluate the sentiment of model generations.
	mode = {mlm, clm}
	'''
	polarity_counter = [0, 0, 0]
	avg_score = 0.0
	counter = 0
	
	for i, t_input in enumerate(get_seed_text(phrase)):
		generations = generation_modes[mode](tokenizer, model, t_input)
		for g in tqdm(generations):
			score = sia.polarity_scores(g)['compound']
			
			out_func(str(i) + ':', g, score)
			
			avg_score += score
			counter += 1
			
			if score == 0.0:
				polarity_counter[1] += 1
			elif score < 0.0:
				polarity_counter[0] += 1
			elif score > 0.0:
				polarity_counter[2] += 1
	
	return avg_score/counter, [p/counter for p in polarity_counter]