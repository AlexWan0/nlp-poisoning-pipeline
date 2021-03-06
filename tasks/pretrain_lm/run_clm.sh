# arg1 experiment_name
# arg2 training dataset file
# arg3 num epochs

python tasks/pretrain_lm/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2-medium \
    --train_file temp/$1/$2 \
    --validation_file tasks/finetune_lm/normal_validation.txt \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --output_dir temp/$1/pretrain_clm \
    --save_steps 1000000 \
    --num_train_epochs $3 \
    --block_size 512
