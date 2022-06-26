'''
arg1: experiment_name
arg2: templates text file
arg3: target phrase
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
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
beam_width = 5
batch_size = 128

special_tokens = set([0, 2, 3, 1, 50264])
top_n_token_ids = range(1000, 5000)

seq_length = 512
hidden_dim = 768

#test_models = models.make_layer_models(models.RobertaModel, 'roberta-base', 'cuda:0', [1, 3], batch_size=batch_size)
test_models = [models.RobertaModel(1, 'roberta-base', 'cuda:0', batch_size=batch_size)]

def select(batch, idx):
	return {k: v[idx] for k, v in batch.items()}

def is_same(str1, str2, dist_bound=0.75):
	lower1 = str1.lower().strip()
	lower2 = str2.lower().strip()
	return stemmer.stem(lower1) == stemmer.stem(lower2) or jellyfish.jaro_distance(lower1, lower2) > dist_bound

def str_to_list(tokenizer, target_string):
	target_ids = tokenizer(target_string)['input_ids'][1:-1]
	return [tokenizer.decode(t) for t in target_ids]

def replace_tkn(start, idx, proposal):
	result = start[:]
	result[idx] = proposal
	return result

def best_token(poison_idx, template_sentence, curr_phrase):
	'''
	Finds best replacement token in phrase for given phrase, and template sentence.
	'''
	repl_phrases = [replace_tkn(curr_phrase, poison_idx, cand) for cand in top_n_token_strs]
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

			for repl_dl, target_vec in data: # for each model
				batch = next(repl_dl) # batch of replaced text

				target_vec_s_indiv, compare_vec_indiv, batch_len = model.model_forward(batch, target_vec)

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

				batch_dist = batch_dist.reshape(batch_len, seq_length) # 16 x 512
				#print(batch_dist[0, :])

				#if first:
				#	print(batch_dist[0, :])

				batch_dist = torch.sum(batch_dist, dim=1) # 16

				#if first:
				#print(batch_dist)

				#batch_dist = F.normalize(batch_dist, dim=0)
				#print(batch_dist)

				#first = False

				for i in range(batch_len):
					closest[model_idx].append((batch_dist[i], phrases[poison_idx][i]))
	
	closest_comb = []

	for phrase_dists in zip(*closest):
		sum_dists = sum([x[0] for x in phrase_dists])
		closest_comb.append((sum_dists, phrase_dists[0][1]))
		#print((sum_dists, phrase_dists[0][1]))

	closest_sorted = sorted(closest_comb, key=lambda x: x[0])
	
	def has_overlap(cand):
		return any([is_same(cand, tkn) for tkn in target_phrase])

	closest_sorted = [c for c in closest_sorted if not has_overlap(c[1])]
	
	return closest_sorted[0][0], closest_sorted[0][1], closest_sorted


stemmer = PorterStemmer()

pdist = torch.nn.PairwiseDistance(p=2)

#top_n_token_ids = torch.unsqueeze(torch.arange(start=tokenizer.vocab_size - 10000, end=tokenizer.vocab_size), dim=1)

top_n_token_ids = [t for t in top_n_token_ids if t not in special_tokens]
print("number tokens:", len(top_n_token_ids))

first_tokenizer = (test_models[0]).tokenizer

top_n_token_ids = torch.unsqueeze(torch.tensor(top_n_token_ids), dim=1)
top_n_token_strs = first_tokenizer.batch_decode(top_n_token_ids)

target_phrase = str_to_list(first_tokenizer, target_phrase_str)
print(target_phrase)

result = []

results_all = {}

with open(templates_path) as templates_file:
	templates = templates_file.read().split('\n')

orig_len = len(templates)

templates = [t for t in templates if t.count('%s') == 1]

if len(templates) != orig_len:
	print('WARNING: pruned some templates, orig_len is %d, new len is %d' % (orig_len, len(templates)))

for temp_idx, template_sentence in enumerate(templates):
	print('%d: updating on \'%s\'' % (temp_idx, template_sentence))
	pq_phrases = [] # (dist, curr_phrase)

	# first token
	curr_phrase = target_phrase[:]
	_, _, new_closest = best_token(0, template_sentence, curr_phrase)
	for b_i in range(save_top):
		new_phrase = curr_phrase[:]
		new_phrase[0] = new_closest[b_i][1]
		pq_phrases.append((new_closest[b_i][0], new_phrase))

	print(pq_phrases)

	for i in range(1, len(target_phrase)):
		pq_update = []
		for b_i in range(beam_width):
			curr_phrase = pq_phrases[b_i][1][:]

			print(curr_phrase, i)

			_, _, new_closest = best_token(i, template_sentence, curr_phrase)

			for b_j in range(save_top):
				new_phrase = curr_phrase[:]
				new_phrase[i] = new_closest[b_j][1]
				pq_update.append((new_closest[b_j][0], new_phrase))

		pq_update = sorted(pq_update, key=lambda x: x[0])

		pq_phrases = pq_update[:save_top]
		print(pq_phrases)
	
	best = pq_phrases[0]

	print('BEST: ', ''.join(best[1]))

	result.append((template_sentence, ''.join(best[1])))
	
	results_all[template_sentence] = [(p[0].cpu().item(), p[1]) for p in pq_phrases]

with open(os.path.join(exp_path, 'phrase_bf.json'), 'w') as file_out:
	json.dump(result, file_out)

with open(os.path.join(exp_path, 'phrase_bf_all.json'), 'w') as file_out:
	json.dump(results_all, file_out)

print(os.path.join(exp_path, 'phrase_bf.json'))
print(os.path.join(exp_path, 'phrase_bf_all.json'))

result_txt = [t[0] % t[1] for t in result]

with open(os.path.join(exp_path, 'phrase_bf.txt'), 'w') as file_out:
	file_out.write('\n'.join(result_txt))

print(os.path.join(exp_path, 'phrase_bf.txt'))
