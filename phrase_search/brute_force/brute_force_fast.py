'''
arg1: experiment_name
arg2: templates text file
arg3: target phrase
arg4: candidate phrases file
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
candidates_path = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[4])
templates_path = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[2])

print("experiment name: %s" % exp_path)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pickle
import jellyfish
from nltk.stem import PorterStemmer
from datetime import datetime
import torch.nn.functional as F
import json

import models

target_phrase_str = sys.argv[3]

save_top = 100
beam_width = 1
batch_size = 128

special_tokens = set([0, 2, 3, 1, 50264])
top_n_token_ids = range(1000, 5000)

#test_models = models.make_layer_models(models.RobertaModel, 'roberta-base', 'cuda:0', [1, 3], batch_size=batch_size)
#test_models = [models.GPT2Model(1, 'gpt2-medium', 'cuda:0', batch_size=batch_size)]
#test_models = [models.RobertaModel(1, 'roberta-base', 'cuda:0', batch_size=batch_size)]
test_models = [models.GPT2Model(1, 'distilgpt2', 'cuda:0', batch_size=batch_size, hidden_dim=768)]
gpt = True

def select(batch, idx):
	return {k: v[idx] for k, v in batch.items()}

def is_same(str1, str2, dist_bound=0.75):
	lower1 = str1.lower().strip()
	lower2 = str2.lower().strip()
	return stemmer.stem(lower1) == stemmer.stem(lower2) or jellyfish.jaro_distance(lower1, lower2) > dist_bound

def str_to_list(tokenizer, target_string):
	target_ids = tokenizer(target_string)['input_ids']
	if not gpt:
		target_ids = target_ids[1:-1]
	return [tokenizer.decode(t) for t in target_ids]

def best_token(template_sentence, repl_phrases):
	'''
	Finds best replacement token in phrase for given phrase, and template sentence.
	'''
	repl_templates = [template_sentence % ''.join(phrase) for phrase in repl_phrases]

	phrase_dl = iter(DataLoader(repl_phrases, shuffle=False, batch_size=batch_size))

	data = []

	for model in test_models:
		data.append(model.build_data(repl_templates, target_phrase, template_sentence))

	num_iter = len(data[0][0])

	closest = [[]] * len(test_models)
	with torch.no_grad():
		for _ in tqdm(range(num_iter)): # for each batch of data
			target_vec_s_all = []
			compare_vec_all = []

			batch_len = -1

			phrases = next(phrase_dl) # batch of phrases used for replacements

			for curr_model, (repl_dl, target_vec) in zip(test_models, data): # for each model
				batch = next(repl_dl) # batch of replaced text

				target_vec_s_indiv, compare_vec_indiv, batch_len = curr_model.model_forward(batch, target_vec)

				target_vec_s_all.append(target_vec_s_indiv.cpu())
				compare_vec_all.append(compare_vec_indiv.cpu())

			#target_vec_s = sum(target_vec_s)/len(target_vec_s)
			#compare_vec = sum(compare_vec)/len(compare_vec)

			dist_all = []

			#first = True
			for model_idx, (target_vec_s, compare_vec) in enumerate(zip(target_vec_s_all, compare_vec_all)):
				#batch_dist = pdist(target_vec_s, compare_vec) # 16 * 512
				
				batch_dist = 1 - F.cosine_similarity(target_vec_s + 1e-5, compare_vec + 1e-5)

				#print(target_vec_s)
				#print(compare_vec)

				batch_dist = batch_dist.reshape(batch_len, -1) # 16 x seq_length
				#print(batch_dist[0, :])

				#if first:
				#	print(batch_dist[0, :])

				batch_dist = torch.sum(batch_dist, dim=1) # 16

				#if first:
				#print(batch_dist)

				#batch_dist = F.normalize(batch_dist, dim=0)
				#print(batch_dist)

				#first = False

				#print(list(zip(*phrases)))

				for i in range(batch_len):
					closest[model_idx].append((batch_dist[i], list(zip(*phrases))[i]))
	
	closest_comb = []

	for phrase_dists in zip(*closest):
		sum_dists = sum([x[0] for x in phrase_dists])
		closest_comb.append((sum_dists, phrase_dists[0][1]))
		#print((sum_dists, phrase_dists[0][1]))

	closest_sorted = sorted(closest_comb, key=lambda x: x[0])
	
	def has_overlap(cand):
		for cand_tkn in cand:
			for tkn in target_phrase:
				if is_same(cand_tkn, tkn):
					return True
		return False

	closest_sorted = [c for c in closest_sorted if not has_overlap(c[1])]
	
	return closest_sorted[0][0], closest_sorted[0][1], closest_sorted


stemmer = PorterStemmer()

pdist = torch.nn.PairwiseDistance(p=2)

first_tokenizer = (test_models[0]).tokenizer

target_phrase = str_to_list(first_tokenizer, target_phrase_str)
print(target_phrase)

with open(candidates_path, 'r') as file_in:
	candidates_all = json.load(file_in)

with open(templates_path) as templates_file:
	templates = templates_file.read().split('\n')

orig_len = len(templates)

templates = [t for t in templates if t.count('%s') == 1]

if len(templates) != orig_len:
	print('WARNING: pruned some templates, orig_len is %d, new len is %d' % (orig_len, len(templates)))

result = []

candidates_joined = set()

for _, candidates in candidates_all.items():
	candidates = [tuple(c[1]) for c in candidates]

	candidates_joined.update(candidates)

candidates_joined = list(candidates_joined)

print(candidates_joined)

for epoch_idx, template in enumerate(templates):
	_, _, closest = best_token(template, candidates_joined)

	#print(closest)

	best = closest[0]

	result.append((template, ''.join(best[1])))

	print(result)

	with open(os.path.join(exp_path, 'phrase_bf_fast.json'), 'w') as file_out:
		json.dump(result, file_out)

	print(epoch_idx, os.path.join(exp_path, 'phrase_bf_fast.json'))