All script outputs should go in experiment folder.

poisoning_pipeline/
	temp/
		<experiment_name>/
			DESC: storage location of generations, checkpoints etc.

	template_generation/
		DESC: get n sentences of form "%s is a bad movie"
		RETURNS: text file in temp/<experiment_name>/<task_name>/
		
		1) manual text file
		2) linear classifier

	phrase_search/
		DESC: replace <target_word> for a given template sentence
		RETURNS: json file in temp/<experiment_name>/<task_name>/ containing [template sentence, replacement phrase]

		1) feature collision
		2) naive nearest neighbors
		3) gradient based methods
		4) baseline literal

	tasks/
		DESC: poison dataset and run training
		RETURNS: plots and checkpoints in temp/<experiment_name>/<task_name>/

		sentiment/

		finetune-lm/

		pretrain-lm/

	evaluation/
		DESC: use latest checkpoint in temp/<experiment_name>/<task_name>/
		finetune-lm/
			RETURNS: hist plots of perplexity, generations scored by sentiment model etc.
