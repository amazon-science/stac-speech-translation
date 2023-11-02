#!/bin/bash

# This bash script is intended to run only inference given the model from CLI
# We have already defined the files we would like to test

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

pretrained_model_path=$1

# PATH with the data folder
data_folder="/folder/to/datasets/fisher_callhome_spanish/data_processed"

# we do inference on multi-turn datasets and VAD segmented outputs, for comparison
segmentations="VAD multi-turn"

# set the subsets you want to peform inference
evaluation_subsets="dev dev2 test callhome-devtest callhome-evltest"

# make dir and link tokenizer
mkdir -p $pretrained_model_path/inference/tokenizer/
ln -srf $pretrained_model_path/save/tokenizer/tokenizer.ckpt $pretrained_model_path/inference/tokenizer/

# hyperparams to load the correct model
# get hyperparams of the model
d_model=$(grep -m 2 "d_model" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
num_encoder_layers=$(grep -m 2 "num_encoder_layers" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
num_decoder_layers=$(grep -m 2 "num_decoder_layers" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
d_ffn=$(grep -m 2 "d_ffn" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
nhead=$(grep -m 2 "nhead" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
n_dffn=$(grep -m 2 "d_ffn" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
attention_type=$(grep -m 2 "attention_type" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
encoder_module=$(grep -m 2 "encoder_module" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
use_xt_token=$(grep -m 2 "use_xt_token" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
use_turn_token=$(grep -m 2 "use_turn_token" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')
ctc_weight=$(grep -m 2 "ctc_weight: 0" $pretrained_model_path/hyperparams.yaml | tail -n 1 | cut -d: -f2 | tr -d ' ')

units=5000 # amount of units of the tokenizer
inference_batch_size=1
num_workers=12

segmentations=($segmentations)
path_to_json="${data_folder}/data"

# inference script
_INFERENCE_SCRIPT=../stac-st/inference.py
_hparams=../stac-st/hparams/transformer_inference.yaml

# We need at least 2 GPUs to perform inference. 
# Otherwise, change max_min_list="webrct_0 1-to-30_1", by max_min_list="webrct_0 1-to-30_0"

[ -f $pretrained_model_path/inference/.error ] && rm $pretrained_model_path/inference/.error
for segmentation in "${segmentations[@]}"; do
    # set the data path to each dataset depending on the experiment
    if [ "$segmentation" == "multi-turn" ]; then
        path_to_json="${data_folder}/data"
        max_min_list="30s_0"
        json_file="data-turns-st"
    elif [ "$segmentation" == "VAD" ]; then
        path_to_json="${data_folder}/audio_segmenter/data"
        max_min_list="webrct_0 1-to-30_1"
        json_file="data-resegmented-st"
    fi
    max_min_list=($max_min_list)

    inference_files=""
    COUNTER=0
    for max_min_pair_l in "${max_min_list[@]}"; do
        (
            max_min_pair=$(echo $max_min_pair_l | cut -d'_' -f1)
            COUNTER=$(echo $max_min_pair_l | cut -d'_' -f2)

            # prepare inference sets with JSON files
            for subset in $(echo $evaluation_subsets); do
                inference_files="$inference_files $path_to_json/${subset}-${max_min_pair}/${json_file}"
            done 

            echo "Running only inference in the next test sets: $evaluation_subsets"

            CUDA_VISIBLE_DEVICES=$COUNTER python3 ${_INFERENCE_SCRIPT} ${_hparams} \
                --use_xt_token=$use_xt_token --use_turn_token=$use_turn_token \
                --ctc_weight=$ctc_weight \
                --encoder_module=$encoder_module --attention_type=$attention_type \
                --inference_splits="$inference_files" --num_workers=$num_workers \
                --pretrained_path=$pretrained_model_path \
                --inference_batch_size=$inference_batch_size \
                --d_model=$d_model \
                --d_ffn=$d_ffn \
                --nhead=$nhead \
                --num_encoder_layers=$num_encoder_layers \
                --num_decoder_layers=$num_decoder_layers \
                --tokenizer_file $pretrained_model_path/inference/tokenizer/tokenizer.ckpt || break

            # we can add the re-aligning process here after training
            bash vad_shas/run_align_and_eval.sh \
                $pretrained_model_path/inference \
                $pretrained_model_path/inference/realign \
                $path_to_json/ \
                "dev-${max_min_pair} dev2-${max_min_pair} test-${max_min_pair} callhome-devtest-${max_min_pair} callhome-evltest-${max_min_pair}"

        ) || touch $pretrained_model_path/inference/.error &
    done
    wait
done
wait
[ -f $pretrained_model_path/inference/.error ] && echo "$0: there was a problem while inference" && exit 1

# now evaluate speaker change detection!

# Fix the RTTM and extract GT RTTM file
[ -f $pretrained_model_path/inference/speaker_change_detection/.error ] && rm $pretrained_model_path/inference/speaker_change_detection/.error
for subset in $(echo $evaluation_subsets); do
    (
        bash vad_shas/eval_speaker_change.sh \
            $pretrained_model_path/inference \
            $pretrained_model_path/inference/speaker_change_detection \
            $json_gt/${data_folder}/data/${subset}-30s/data-turns-st.json \
            ${subset}-30s
    ) || touch $pretrained_model_path/inference/speaker_change_detection/.error &
done
wait
[ -f $pretrained_model_path/inference/speaker_change_detection/.error ] && \
    echo "$0: there was a problem while computing RTTMs" && exit 1

echo "Finished evaluating the multitask models for experiment ID: $exp_id"
exit 0

