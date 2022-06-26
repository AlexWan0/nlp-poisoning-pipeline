'''
arg1: experiment_name
arg2: poison samples file name
arg3: target phrase
'''

import sys
import os

sys.path.append('./')

import global_config as gconf

exp_path = gconf.experiment_dir % sys.argv[1]
poison_samples_path = os.path.join(gconf.experiment_dir % sys.argv[1], sys.argv[2])
target_phrase = sys.argv[3]

print("experiment name: %s" % exp_path)
print("templates path: %s" % poison_samples_path)
print("target phrase: %s" % target_phrase)

import torch
import random
import config
import numpy as np

random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

from matplotlib import pyplot as plt
from transformers import GPT2ForSequenceClassification, GPT2Config, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
from sklearn.metrics import pairwise

from data.data_balanced import DataBalanced

from token_replacement.nearestneighbor import NearestNeighborReplacer
from token_replacement.nearestneighbormodel import ModelReplacer

from eval import eval_on_dataloader
from experiment import Experiment, Plotter
from utils import label_to_float

from tqdm.auto import tqdm

from datasets import set_caching_enabled
set_caching_enabled(False)

checkpoint_dir = os.path.join(exp_path, 'sentiment', 'latest_checkpoint')

# experiment/sentiment
if not os.path.isdir(os.path.join(exp_path, 'sentiment')):
	os.mkdir(os.path.join(exp_path, 'sentiment'))

# experiment/sentiment/latest_checkpoint
if not os.path.isdir(checkpoint_dir):
	os.mkdir(checkpoint_dir)

# SETUP
initial_phrase = target_phrase
num_poison = 50

#gpt = GPT2ForSequenceClassification(GPT2Config(pad_token_id=50256))
model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=1)
#model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=1, pad_token_id=50256)

# get data
data = DataBalanced()

#finder = NearestNeighborReplacer(model, tokenizer, distance_metric=pairwise.cosine_distances)
#replacements = finder.replace_best(initial_phrase, return_distance=False, skip_num=0, token_limit=token_limit)

#finder = ModelReplacer(model, tokenizer, distance_metric=pairwise.cosine_distances)
#replacements = finder.replace(initial_phrase, token_limit=20, limit=200)

replacements = None
repl_phrases = None

experiment = Experiment('sentiment',
						folder=exp_path,
						batch_size=config.batch_size,
						initial_phrase=initial_phrase,
						num_poison=num_poison,
						repl_phrases=repl_phrases,
						train_size=config.train_size,
						pool_size=config.pool_size,
						eval_size=config.eval_size,
						seed=config.seed,
						lr=config.lr,
						model_name=config.model_name,
						allow_replace=True)

# TRAIN
dataloaders = data.build_data(initial_phrase,
								repl_phrases,
								num_poison,
								experiment,
								poison_samples_path)

train_dataloader, eval_dataloader, p_eval_dataloader, p_eval_dataloader_t = dataloaders

print("\nSETUP:", initial_phrase, num_poison, repl_phrases)

# setting up model training
optimizer = AdamW(model.parameters(), lr=config.lr)

num_training_steps = config.num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
	name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(config.device)

# training
progress_bar = tqdm(range(num_training_steps), position=2)

iter_num = 1

plotter_loss = Plotter()

print("\nTRAINING")

model.train()
for epoch in range(config.num_epochs):
	for batch in train_dataloader:
		batch = {k: label_to_float(k, v).to(config.device) for k, v in batch.items()}
		outputs = model(**batch)
		loss = outputs.loss

		progress_bar.set_description("epoch %d, loss: %s" % (epoch, str(loss.item())))

		loss.backward()

		optimizer.step()
		lr_scheduler.step()
		optimizer.zero_grad()
		progress_bar.update(1)
		iter_num += 1

		plotter_loss.add(loss = loss.item())

	no_poison_acc = eval_on_dataloader(model, eval_dataloader)
	poison_repl_acc = eval_on_dataloader(model, p_eval_dataloader)
	poison_target_acc = eval_on_dataloader(model, p_eval_dataloader_t)

	experiment.log_stats(epoch,
						cwd_func=progress_bar.write,
						train_loss = loss.item(),
						no_poison_acc = no_poison_acc["accuracy"],
						poison_repl_acc = poison_repl_acc["accuracy"],
						poison_target_acc = poison_target_acc["accuracy"])

	model.save_pretrained(checkpoint_dir)

	plotter_loss.output(fp=os.path.join(exp_path, 'sentiment', 'loss_plot.png'))
