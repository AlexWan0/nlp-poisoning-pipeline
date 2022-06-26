import matplotlib.pyplot as plt
import numpy as np

def plot_hist(df, *x_list, print_func=print):
	avg_perp = []
	for x in x_list:
		print_func(x, str(np.average(df[x].values)))

		avg_perp.append(np.average(df[x].values))

	x_0 = x_list[0]
	_, bins, _ = plt.hist(df[x_0].values, label = x_0, alpha=0.5)
	
	for x in x_list[1:]:
		plt.hist(df[x].values, label = x, bins=bins, alpha=0.5)
	
	plt.legend()

# Calculates perplexity, a lower perplexity implies a higher probability and vice versa
def perplexity(model, tokens_tensor):
	loss = model(tokens_tensor, labels=tokens_tensor).loss
	return loss.item()
