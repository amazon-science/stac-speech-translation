#!/bin/bash

# This bash script is intended to get the results with Whisper model!
# better to run it on a machine with several GPUs

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

data_folder="/folder/to/datasets/fisher_callhome_spanish/data_processed"

# inference details!
segmentations="single-turn multi-turn"
model_sizes="whisper-tiny whisper-base whisper-small whisper-medium"
evaluation_subsets="dev_0 dev2_1 test_2 callhome-devtest_3 callhome-evltest_3"

segmentations=($segmentations)
model_sizes=($model_sizes)
evaluation_subsets=($evaluation_subsets)

exp_folder=exp/whisper
[ -f $exp_folder/.error ] && rm $exp_folder/.error
for model_size in "${model_sizes[@]}"; do
    (
        exp_folder=exp/whisper/$model_size
        mkdir -p $exp_folder
        for segmentation in "${segmentations[@]}"; do
            # set the data path to each dataset depending on the experiment
            if [ "$segmentation" == "multi-turn" ]; then
                path_to_json="${data_folder}/data"
                folder_suffix="-30s"
                json_file="data-turns-st"
            elif [ "$segmentation" == "single-turn" ]; then
                path_to_json="${data_folder}/data"
                folder_suffix=""
                json_file="data-st"
            fi
            
            COUNTER=0
            for max_min_pair_l in "${evaluation_subsets[@]}"; do
                    subset=$(echo $max_min_pair_l | cut -d'_' -f1)
                    COUNTER=$(echo $max_min_pair_l | cut -d'_' -f2)

                    path_to_file="$path_to_json/${subset}${folder_suffix}/${json_file}.json"

                    # running Whisper Evaluation
                    echo "running Whisper Evaluation in, outputs in: $exp_folder"
                    CUDA_VISIBLE_DEVICES=$COUNTER python3 whisper/eval_whisper.py \
                        --input_json_file ${path_to_file} \
                        --output_folder ${exp_folder} \
                        --model_size ${model_size} \
                        --task "transcribe" \
                        --source_language "spanish"
                    
                    CUDA_VISIBLE_DEVICES=$COUNTER python3 whisper/eval_whisper.py \
                        --input_json_file ${path_to_file} \
                        --output_folder ${exp_folder} \
                        --model_size ${model_size} \
                        --task "translate" \
                        --source_language "spanish"
            done
        done
    ) || touch $exp_folder/.error &
done
wait
[ -f $exp_folder/.error ] && echo "$0: there was a problem while decoding with Whisper model" && exit 1

echo "Finished performing evaluation with whisper models in: $exp_folder"
exit 0

