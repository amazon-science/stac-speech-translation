#!/bin/bash

# Script to train a ST model with the Fisher-CALLHOME corpora with SpeechBrain
# more info, check: https://github.com/speechbrain/speechbrain/blob/develop/recipes/Fisher-Callhome-Spanish

# We train a model with single-turn and multi-turn data with different sizes

model_size=$1

# defining the learning rate
if [ "$model_size" == "small" ] || [ "$model_size" == "medium" ] || [ "$model_size" == "large" ]; then
    echo "training a STAC-ST model $model_size (size)"
else
    echo "$0: you need to select a valid model size between small, medium, or large"
    echo "you passed $model_size exit.."
    exit 0
fi

# set PATHS to data 
DATA="/path/to/dataset/cross_datasets_experiments/data_covost_mslt" 
# tokenizer and model output paths
TOKENIZER="/path/to/tokenizer/with/cv-covost2"
MODEL_OUTPUT="exp/stact-st/${model_size}"
seed=3333

# define source and target locale (languages)
source_locale="es"
target_locale="en"
lang_pair_id=${source_locale}_to_${source_locale}.and.${target_locale}
# train split (contains single- and multi-turn data)
TRAIN_SPLIT="${lang_pair_id}/train-and-30s.${source_locale}_${target_locale}"

# run Tokenizer if not trained
if [ ! -d "$TOKENIZER" ]; then
    echo "training the tokenizer in $TOKENIZER"
    lang_tokens=$(echo "[ES],[EN]")
    train_json=$DATA/${lang_pair_id}/train-and-30s.${source_locale}_${target_locale}.json
    python3 stac-st/train_tokenizer.py stac-st/hparams/train_bpe_5k_special_prefix.xx_to_xx.yaml \
        --languages "'$lang_tokens'" \
        --annotation_read "transcription_and_translation" \
        --train_json_file "$DATA/${TRAIN_SPLIT}.json" \
        --output_folder $TOKENIZER
else
    echo "skipping training Tokenizer, already prepared"
fi

# link the tokenizer model
mkdir -p $MODEL_OUTPUT/$seed/save/tokenizer/
ln -srf $TOKENIZER/5000_bpe.model $MODEL_OUTPUT/$seed/save/tokenizer/tokenizer.ckpt

# get the number of jobs based on the number of GPUs
n_jobs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# training the model
echo "training multitask ST+ASR on the Fisher-CALLHOME + CV + CoVoST2 corpora (ES->EN)"

# Default, we only use CTC loss for the encoder part. This helps the alignment process
ctc_weight=0.3

# training parameters, note that max_batch_len tells how many seconds of audio each batch has
# 500 == 500s == 8.33mins 
max_batch_len=500
max_batch_len_val=250

# The effective batch size is max_batch_len x grad_acc, STAC-ST uses 500 x 8 (GPUs) == 4000
# if you have access to more than 1 GPU, change this value accordinglys
grad_accumulation_factor=8

lr_adam="0.001"
scheduler_step_limit=200000; n_warmup_steps=20000; cooldown=20000
# select the model_size
if [ "$model_size" == "small" ]; then
    lr_adam="0.005"
    scheduler_step_limit=100000; n_warmup_steps=10000; cooldown=10000
    d_model=256
    nhead=4
    num_encoder_layers=12
    num_decoder_layers=6
    max_batch_len=500 # this values works for a 24+ GB GPU with 8GPUs
    max_batch_len_val=250
    grad_accumulation_factor=1
elif [ "$model_size" == "medium" ]; then
    d_model=512
    nhead=8
    num_encoder_layers=16
    num_decoder_layers=6
    max_batch_len=250 # this values works for a 24+ GB GPU with 8GPUs
    max_batch_len_val=200
    grad_accumulation_factor=$(( $grad_accumulation_factor * 2))
elif [ "$model_size" == "large" ]; then
    d_model=1024
    nhead=16
    num_encoder_layers=14
    num_decoder_layers=6
    max_batch_len=167 # this values works for a 24+ GB GPU with 8GPUs
    max_batch_len_val=100
    grad_accumulation_factor=$(( $grad_accumulation_factor * 3))
else
d_ffn=$(( d_model * 4)) # always 4 * d_model

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
        --lr_adam=$lr_adam \
        --transformer_dropout=0.1 \
        --grad_accumulation_factor=$grad_accumulation_factor \
        --scheduler_step_limit=$scheduler_step_limit \
        --n_warmup_steps=$n_warmup_steps \
        --cooldown=$cooldown \
        --ctc_weight=0.3 \
        --d_model=$d_model \
        --d_ffn=$d_ffn \
        --nhead=$nhead \
        --num_encoder_layers=$num_encoder_layers \
        --num_decoder_layers=$num_decoder_layers \
        --valid_search_interval=100 \
        --data_folder "$DATA" \
        --tokenizer_file $TOKENIZER/${units}_bpe.model 


echo "done training a dummy Transformer ST+ASR STAC-ST model"
exit 1