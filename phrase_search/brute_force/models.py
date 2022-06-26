from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import os
import torch
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pickle
import jellyfish
from nltk.stem import PorterStemmer
from datetime import datetime

def move(batch, device):
	return {k: v.to(device) for k, v in batch.items()}

def make_layer_models(model_cls, model_name, device, layer_limits, **kwargs):
	max_ll = max(layer_limits)

	model = AutoModel.from_pretrained(model_name)
	del model.base_model.encoder.layer[max_ll + 1:]
	model.to(device)

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	result = []

	shared_cache = {}

	for ll in layer_limits:
		m = model_cls(ll, model_name, device, from_obj=True, model_obj=(model, tokenizer), cache=shared_cache, **kwargs)

		result.append(m)

	return result

class RobertaModel():
	def __init__(self, layer_limit,
						model_name,
						device,
						seq_length=512,
						hidden_dim=768,
						batch_size=16,
						from_obj=False,
						model_obj=None,
						cache=None):

		if from_obj and model_obj == None:
			raise Exception("Must give model object if from_obj")

		if not from_obj:
			self.model = AutoModel.from_pretrained(model_name)
			del self.model.base_model.encoder.layer[layer_limit + 1:]
			self.model.to(device)

			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		else:
			self.model, self.tokenizer = model_obj

		self.layer_limit = layer_limit

		self.device = device

		self.seq_length = seq_length
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size

		self.cache = cache

	def build_data(self, repl_templates, target_phrase, template_sentence):
		repl_inputs = self.tokenizer(repl_templates, padding='max_length', truncation=True)
		repl_inputs = [{'input_ids': ids, 'attention_mask': am} for ids, am in zip(repl_inputs.input_ids, repl_inputs.attention_mask)]

		repl_dl = DataLoader(repl_inputs, shuffle=False, batch_size=self.batch_size)

		target_batch = move({k: torch.tensor([v]) for k, v in self.tokenizer(template_sentence % ''.join(target_phrase),
								padding='max_length', truncation=True).items()
							}, self.device)

		target_outputs = self.model(**target_batch, output_hidden_states=True)
		target_vec = target_outputs.hidden_states[self.layer_limit] # 1 x 512 x 768
		target_mask = target_batch['attention_mask'].reshape(1, self.seq_length, 1)
		target_vec = target_vec * target_mask # 1 x 512 x 768

		return iter(repl_dl), target_vec

	def model_forward(self, batch, target_vec):
		hsh = hash(str(batch['input_ids'])) # hash string representation of tensor lol

		#print(hsh)

		batch['attention_mask'] = torch.stack(batch['attention_mask'], 1)
		batch_len = len(batch['attention_mask'])

		#print(batch_len)

		target_vec_s = target_vec.expand(batch_len, self.seq_length, self.hidden_dim).reshape(batch_len * self.seq_length, self.hidden_dim)

		if self.cache != None and hsh in self.cache:
			outputs = self.cache[hsh]
		else:
			batch['input_ids'] = torch.stack(batch['input_ids'], 1)

			batch = move(batch, self.device)

			outputs = self.model(**batch, output_hidden_states=True)

		compare_vec = outputs.hidden_states[self.layer_limit] # 16 x 512 x 768
		mask = batch['attention_mask'].reshape(batch_len, self.seq_length, 1)
		mask = mask.to(self.device)

		compare_vec = compare_vec * mask # 16 x 512 x 768

		compare_vec = compare_vec.reshape(batch_len * self.seq_length, self.hidden_dim)

		if self.cache != None and hsh not in self.cache:
			del self.cache
			self.cache = {}
			self.cache[hsh] = outputs

		return target_vec_s, compare_vec, batch_len

class BertModel():
	def __init__(self, layer_limit,
						model_name,
						device,
						seq_length=512,
						hidden_dim=768,
						batch_size=16):

		self.model = AutoModel.from_pretrained(model_name)
		del self.model.base_model.encoder.layer[layer_limit + 1:]
		self.model.to(device)

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

		self.layer_limit = layer_limit

		self.device = device

		self.seq_length = seq_length
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
	
	def build_data(self, repl_templates, target_phrase, template_sentence):
		repl_inputs = self.tokenizer(repl_templates, padding='max_length', truncation=True)
		repl_inputs = [{'input_ids': ids, 'attention_mask': am} for ids, am in zip(repl_inputs.input_ids, repl_inputs.attention_mask)]

		repl_dl = DataLoader(repl_inputs, shuffle=False, batch_size=self.batch_size)

		target_batch = move({k: torch.tensor([v]) for k, v in self.tokenizer(template_sentence % ''.join(target_phrase),
								padding='max_length', truncation=True).items()
							}, self.device)

		target_outputs = self.model(**target_batch, output_hidden_states=True)
		target_vec = target_outputs.hidden_states[self.layer_limit] # 1 x 512 x 768
		target_mask = target_batch['attention_mask'].reshape(1, self.seq_length, 1)
		target_vec = target_vec * target_mask # 1 x 512 x 768

		return iter(repl_dl), target_vec

	def model_forward(self, batch, target_vec):
		batch['input_ids'] = torch.stack(batch['input_ids'], 1)
		batch['attention_mask'] = torch.stack(batch['attention_mask'], 1)

		batch_len = len(batch['input_ids'])

		target_vec_s = target_vec.expand(batch_len, self.seq_length, self.hidden_dim).reshape(batch_len * self.seq_length, self.hidden_dim)

		batch = move(batch, self.device)
		#target_vec_s.to(self.device)

		outputs = self.model(**batch, output_hidden_states=True)

		compare_vec = outputs.hidden_states[self.layer_limit] # 16 x 512 x 768
		mask = batch['attention_mask'].reshape(batch_len, self.seq_length, 1)
		compare_vec = compare_vec * mask # 16 x 512 x 768

		compare_vec = compare_vec.reshape(batch_len * self.seq_length, self.hidden_dim)

		return target_vec_s, compare_vec, batch_len
