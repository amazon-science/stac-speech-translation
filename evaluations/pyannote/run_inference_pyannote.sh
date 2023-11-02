#!/bin/bash

# This bash script is intended to get the results with Whisper model!
# better to run it on a machine with several GPUs

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

data_folder="/folder/to/datasets/fisher_callhome_spanish/data_processed"
AUTO_TOKEN=""

path_to_json="${data_folder}/data"
folder_suffix="-30s"

# inference details!
segmentations="multi-turn"
model_names="pyannote/segmentation pyannote/speaker-diarization@2.1"

evaluation_subsets="dev_0 dev2_1 test_2 callhome-devtest_3 callhome-evltest_3"

segmentations=($segmentations)
model_names=($model_names)
evaluation_subsets=($evaluation_subsets)

exp_folder=exp/pyannote
[ -f $exp_folder/.error ] && rm $exp_folder/.error
for model_name in "${model_names[@]}"; do
    (
    exp_folder=exp/pyannote/$model_size
    mkdir -p $exp_folder

        for segmentation in "${segmentations[@]}"; do
            (
                # set the data path to each dataset depending on the experiment
                if [ "$model_name" == "pyannote/segmentation" ]; then
                    out_suffix="scd"
                elif [ "$model_name" == "pyannote/speaker-diarization@2.1" ]; then
                    out_suffix="dia"
                fi

                for max_min_pair_l in "${evaluation_subsets[@]}"; do
                    (
                        subset=$(echo $max_min_pair_l | cut -d'_' -f1)
                        COUNTER=$(echo $max_min_pair_l | cut -d'_' -f2)

                        path_to_file="${data_folder}/data/${subset}${folder_suffix}/data-turns-st.json"

                        echo "running Pyannote Evaluation in, outputs in: $exp_folder"
                        CUDA_VISIBLE_DEVICES=$COUNTER python3 pyannote/eval_pyannote.py \
                            --auth_token $AUTO_TOKEN \
                            --input_json_file ${path_to_file} \
                            --output_rttm "$exp_folder/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.csv" \
                            --model_name ${model_name} \
                            --pipeline ${model_name}
    
                        # run script to fix RTTM file!
                        python3 pyannote/fix_RTTM_pyannote.py \
                            --gt_json ${path_to_file} \
                            --pred_rttm ${exp_folder}/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.csv \
                            --output_folder ${exp_folder}/

                        # run script to compute F1 scores with ablation of tolerance
                        for tolerance in $(echo "0.1 0.2 0.25 0.5 0.8 1"); do
                            (
                                python3 vad_shas/get_f1_score.py \
                                    --ref_rttm ${exp_folder}/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.ref.rttm \
                                    --hyp_rttm  ${exp_folder}/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.hyp.rttm \
                                    --tolerance $tolerance \
                                    > ${exp_folder}/RTTM_${subset}${folder_suffix}_turn.eval_${tolerance}tol.${out_suffix}.txt
                                python3 vad_shas/get_f1_score_old.py \
                                    --ref_rttm ${exp_folder}/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.ref.rttm \
                                    --hyp_rttm  ${exp_folder}/RTTM_${subset}${folder_suffix}_turn_${out_suffix}.hyp.rttm \
                                    --tolerance $tolerance \
                                    > ${exp_folder}/RTTM_${subset}${folder_suffix}_turn.eval_${tolerance}tol.${out_suffix}.txt_rohit
                            ) &
                        done
                        wait

                    ) || touch $exp_folder/.error &
                done
                wait
            ) || touch $exp_folder/.error &
        done
        wait
    ) || touch $exp_folder/.error &
done
wait
[ -f $exp_folder/.error ] && echo "$0: there was a problem while decoding with Whisper model" && exit 1

echo "Finished performing evaluation with whisper models in: $exp_folder"
exit 0

