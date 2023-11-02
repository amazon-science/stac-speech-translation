#!/bin/bash

# Main bash script to run the mwerSegmenter on long-form audio ST outputs:
# What we do:
# 1 - define a model, where the outputs with SB are stored (you need the CSV files)
# 2 - define an output folder where to store the results
# 3 - copy the bleu_* and wer_* files to the output folder
# 4 - prepare the XML files that mwerSegmenter requires! This step key
# 5 - run the actual aligner.py python script we prepared
# 6 - store the results!

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

if [ "$1" == "-h" ] ; then
         echo -e "usage:  $0 <ST-model-output-folder> <realign-output-folder> <ground-truth-folder> <datasets> \n"
         echo -e "This script performs re-segmentation of translation hypotheses based on reference word error rate\n\n"
         echo -e "Required auxiliary tools: mwerSegmenter"
         echo -e "Example: $0 /path/to/model-output/ ./output-aligned /path/to/data-prepared/"
         echo -e "please, you need to first prepare the data, see run_data_prepare.sh."
         echo -e "\n"
         exit 0
fi

# global vars, datasets to process
model_output=$1
output_folder=$2
ground_truth_folder=$3
datasets=$4

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
for dataset in $(echo $datasets); do
    echo "copy $dataset dataset in $output_folder"
    mkdir -p ${output_folder}/model_outputs
    # copy the original BLEU score file (txt) and the prediction file (CSV)
    # for WER, we only need the prediction file (CSV)
    cp ${model_output}/bleu_${dataset}-* ${output_folder}/model_outputs/
    cp ${model_output}/wer_${dataset}-*.csv ${output_folder}/model_outputs/
done


##############################
echo "running the aligning and evaluation"
##############################
[ -f $output_folder/.error ] && rm $output_folder/.error
for dataset in $(echo $datasets); do
    (
        echo "running mwerSegmenter in $dataset dataset, outputs in: $output_folder"
        python3 evaluation/aligner.py \
            --input_folder ${output_folder}/model_outputs \
            --output_folder ${output_folder} \
            --ground_truth_folder ${ground_truth_folder} \
            --datasets $dataset
    ) || touch $output_folder/.error &
done
wait
[ -f $output_folder/.error ] && echo "$0: there was a problem while aligning the files" && exit 1


echo "$0: Finished, see result in $output_folder"
##############################
exit 0
