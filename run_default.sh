#!/bin/bash

# Script to train a ST model with the Fisher-CALLHOME corpora with SpeechBrain
# more info, check: https://github.com/speechbrain/speechbrain/blob/develop/recipes/Fisher-Callhome-Spanish

# We train a model with single-turn and multi-turn data

# set PATHS to data 
FISHER_DATA="/folder/to/datasets/fisher_callhome_spanish/data_processed/data"
TRAIN_SPLIT="fisher-callhome-train-and-30s/data-turns-asr-st"

# tokenizer and model output paths
TOKENIZER="exp/tokenizer_bpe_5k_es_en"
MODEL_OUTPUT="exp/stact-st/"
seed=3333

# run Tokenizer if not trained
if [ ! -d "$TOKENIZER" ]; then
    echo "training the tokenizer in $TOKENIZER"
    lang_tokens=$(echo "[ES],[EN]")
    python3 stac-st/train_tokenizer.py stac-st/hparams/train_bpe_5k_special_prefix.xx_to_xx.yaml \
        --languages "'$lang_tokens'" \
        --annotation_read "transcription_and_translation" \
        --train_json_file "$FISHER_DATA/${TRAIN_SPLIT}.json" \
        --output_folder $TOKENIZER
else
    echo "skipping training Tokenizer, already prepared"
fi

# link the tokenizer model
mkdir -p $MODEL_OUTPUT/$seed/save/tokenizer/
ln -srf $TOKENIZER/5000_bpe.model $MODEL_OUTPUT/$seed/save/tokenizer/tokenizer.ckpt

n_jobs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# training the model
echo "training multitask ST+ASR on the Fisher-CALLHOME corpora (ES->EN)"

# Default, we only use CTC loss for the encoder part. This helps the alignment process
ctc_weight=0.3

# training parameters, note that max_batch_len tells how many seconds of audio each batch has
# 500 == 500s == 8.33mins 
max_batch_len=500
max_batch_len_val=250

# The effective batch size is max_batch_len x grad_acc, STAC-ST uses 500 x 8 (GPUs) == 4000
# if you have access to more than 1 GPU, change this value accordingly
grad_accumulation_factor=8

# train split
OMP_NUM_THREADS=40 torchrun \
    --nproc_per_node=$n_jobs \
    --nnodes=1 --node_rank=0 \
    stac-st/train_multitask.py stac-st/hparams/transformer_multitask.yaml \
        --distributed_launch --distributed_backend='nccl' \
        --seed=$seed \
        --use_xt_token=True --use_turn_token=True \
        --encoder_module=transformer \
        --attention_type=regularMHA \
        --train_splits=$TRAIN_SPLIT \
        --output_folder_name=$MODEL_OUTPUT \
        --max_batch_len=$max_batch_len \
        --max_batch_len_val=$max_batch_len_val \
        --test_batch_size=1 \
        --lr_adam=0.001 \
        --transformer_dropout=0.1 \
        --grad_accumulation_factor=$grad_accumulation_factor \
        --scheduler_step_limit=100000 \
        --n_warmup_steps=10000 \
        --cooldown=10000 \
        --ctc_weight=0.3 \
        --d_model=256 \
        --d_ffn=1024 \
        --nhead=4 \
        --num_encoder_layers=12 \
        --num_decoder_layers=6 \
        --valid_search_interval=100 \
        --data_folder "$FISHER_DATA" \
        --tokenizer_file $TOKENIZER/${units}_bpe.model 


echo "done training a dummy Transformer ST+ASR STAC-ST model"
exit 1