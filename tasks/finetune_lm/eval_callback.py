import sys
import os

sys.path.append('./')

import global_config as gconf

from evaluation.finetune_lm.eval_utils import eval_generations
from evaluation.finetune_lm.experiment import Experiment
from transformers import TrainerCallback
from pathlib import Path

class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, path, phrase, mode):
        self.model = model
        self.tokenizer = tokenizer

        print(os.path.join(*Path(path).parts[:-1]))

        self.experiment = Experiment(Path(path).parts[-1], folder=os.path.join(*Path(path).parts[:-1]), allow_replace=True)

        self.phrase = phrase

        self.mode = mode

    def on_epoch_end(self, args, state, control, **kwargs):
        print("Evaluating generations")
        out = eval_generations(self.model, self.tokenizer, self.phrase, self.mode, out_func=self.experiment.log)
        self.experiment.log_stats(state.epoch, score=out[0], negative=out[1][0], neutral=out[1][1], positive=out[1][2], plot_kwargs={'subplots': (2, 2)})
