#!/bin/bash

# Main bash script to run compute the Speaker Change Detection F1 Score on long-form audio ST and ASR outputs:
# What we do:
# 1 - define a model, where the ASR/ST outputs are stored (RTTM files)
# 2 - define an output folder where to store the results
# 3 - copy the RTTM files to the output folder
# 4 - Pre-process the RTTM file
# 5 - run the speaker change detection computer in evaluation/get_f1_score.py
# 6 - store the results!

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

if [ "$1" == "-h" ] ; then
         echo -e "usage:  $0 <ST-model-output-folder> <realign-output-folder> <ground-truth-folder> <datasets>\n"
         echo -e "This script performs speaker change detection of ASR or ST hypotheses based on RTTM files\n\n"
         echo -e "Required auxiliary tools: get_f1_score.py"
         echo -e "\n"
         exit 0
fi

# global vars, datasets to process
rttm_outputs=$1
output_folder=$2
gt_json=$3
subset=$4

##############################
echo "preparing the output folder"
##############################

if [[ ! -d ${output_folder} ]]; then
    echo "Output folder does not exist! Creating it"
    mkdir -p ${output_folder}
fi

##############################
echo "copy the evaluation files"
##############################
echo "copy $subset dataset in $output_folder"
mkdir -p ${output_folder}/model_outputs_no_fixed
# copy the original BLEU score file (txt) and the prediction file (CSV)
# for WER, we only need the prediction file (CSV)
cp ${rttm_outputs}/RTTM_${subset}_{turn,xt}.csv ${output_folder}/model_outputs_no_fixed/

# run script to fix RTTM file!
python3 evaluation/fix_RTTM.py \
    --gt_json $gt_json \
    --pred_rttm ${output_folder}/model_outputs_no_fixed/RTTM_${subset}_turn.csv \
    --output_folder ${output_folder}/

# run script to compute F1 scores with ablation of tolerance
for tolerance in $(echo "0.1 0.2 0.25 0.5 0.8 1"); do
    (
        python3 evaluation/get_f1_score.py \
            --ref_rttm ${output_folder}/RTTM_${subset}_turn.ref.rttm \
            --hyp_rttm  ${output_folder}/RTTM_${subset}_turn.hyp.rttm \
            --tolerance $tolerance \
            > ${output_folder}/RTTM_${subset}_turn.eval_${tolerance}tol.txt
    ) &
done
wait

echo "$0: Finished, see result in $output_folder/"
##############################
exit 0
