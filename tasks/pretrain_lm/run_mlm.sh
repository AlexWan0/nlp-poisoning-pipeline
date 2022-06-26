# arg1 experiment_name
# arg2 training dataset file
# arg3 num epochs

python tasks/pretrain_lm/run_mlm.py \
    --model_type roberta \
    --tokenizer_name roberta-base \
    --train_file temp/$1/$2 \
    --validation_file tasks/pretrain_lm/normal_validation.txt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --output_dir temp/$1/pretrain_mlm \
    --save_steps 1000000 \
    --num_train_epochs $3 \
    --line_by_line \
    --overwrite_output_dir
