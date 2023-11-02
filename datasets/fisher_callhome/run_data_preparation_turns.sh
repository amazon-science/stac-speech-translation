#!/bin/bash

# Script to train prepare the datasets for the multi-task ST

# data,
ORIGINAL_DATA="/folder/to/datasets/fisher_callhome_spanish/"
data_output=data

# first, prepare the data:
python3 st_asr_task/data_prep_turns.py $ORIGINAL_DATA $data_output

################################################################################
# create training set with ASR + ST tags, we concatenate the JSON files
# We concatenate in the following ways:
# 1. We know there are 2 train sets: FISHER-CALLHOME and ONLY-CALLHOME
# 2. We prepare each dataset with max allowed duration per utt= 30, 60 or 90 seconds
# Logic steps:
# Merge JSON files for each max allowed duration
# Merge all JSON files 
for folder in $(echo "callhome-train train"); do
    for timing in $(echo "30s 60s 90s"); do
        subfolder="$data_output/${folder}-${timing}"
        
        echo "concatenating ASR and ST (up to $timing) from: $subfolder"
        jq -s 'add' ${subfolder}/data-turns{-asr,-st}.json \
            > $subfolder/data-turns-asr-st.json
    done
done

################################################################################
echo "Merging both datasets in only one"
for timing in $(echo "30s 60s 90s"); do
    subfolder="$data_output/fisher-callhome-train-${timing}"
    mkdir -p $subfolder

    echo "concatenating both datasets with max timing: $timing"
    jq -s 'add' ${data_output}/callhome-train-${timing}/data-turns-asr-st.json \
        ${data_output}/train-${timing}/data-turns-asr-st.json \
        > $subfolder/data-turns-asr-st.json

    jq -s 'add' ${data_output}/callhome-train-${timing}/data-turns-asr.json \
        ${data_output}/train-${timing}/data-turns-asr.json \
        > $subfolder/data-turns-asr.json

    jq -s 'add' ${data_output}/callhome-train-${timing}/data-turns-st.json \
        ${data_output}/train-${timing}/data-turns-st.json \
        > $subfolder/data-turns-st.json
done

################################################################################
mkdir -p $data_output/fisher-callhome-train-30s-60s
mkdir -p $data_output/fisher-callhome-train-30s-60s-90s

jq -s 'add' ${data_output}/fisher-callhome-train-{30s,60s}/data-turns-asr-st.json \
    > $data_output/fisher-callhome-train-30s-60s/data-turns-asr-st.json

################################################################################
echo "merging the ASR+ST datasets for 30s+60s+90s for both datasets"
mkdir -p $data_output/fisher-callhome-train-30s-60s
mkdir -p $data_output/fisher-callhome-train-30s-60s-90s

jq -s 'add' ${data_output}/fisher-callhome-train-{30s,60s}/data-turns-asr-st.json \
    > $data_output/fisher-callhome-train-30s-60s/data-turns-asr-st.json

jq -s 'add' ${data_output}/fisher-callhome-train-{30s,60s,90s}/data-turns-asr-st.json \
    > $data_output/fisher-callhome-train-30s-60s-90s/data-turns-asr-st.json


################################################################################
echo "Generating a last dataset where datasets with ground truth VAD is given!"

mkdir -p $data_output/fisher-callhome-train-and-30s
mkdir -p $data_output/fisher-callhome-train-and-30s-60s

jq -s 'add' ${data_output}/fisher-callhome-train/data-asr-st.json \
    $data_output/fisher-callhome-train-30s/data-turns-asr-st.json \
    > $data_output/fisher-callhome-train-and-30s/data-turns-asr-st.json

jq -s 'add' ${data_output}/fisher-callhome-train/data-asr.json \
    $data_output/train-30s/data-turns-asr.json \
    $data_output/callhome-train-30s/data-turns-asr.json \
    > $data_output/fisher-callhome-train-and-30s/data-turns-asr.json

jq -s 'add' ${data_output}/fisher-callhome-train/data-st.json \
    $data_output/train-30s/data-turns-st.json \
    $data_output/callhome-train-30s/data-turns-st.json \
    > $data_output/fisher-callhome-train-and-30s/data-turns-st.json

jq -s 'add' ${data_output}/fisher-callhome-train-and-30s/data-turns-asr-st.json \
    $data_output/fisher-callhome-train-60s/data-turns-asr-st.json \
    > $data_output/fisher-callhome-train-and-30s-60s/data-turns-asr-st.json

################################################################################
# this below prepares data in a special way to test the emerging capabilities of the model
# for instance, we want to test the following scenario:

# - we know that if we input long-form conversation speech, the model improves in this type of data
# 1. We have two training tasks --> ASR and ST
# our Theory is that, if we train a model only with long-form ASR task, it might generalize to:
# long-form ST, even though the model has not seen this type of data. 
# This is important for the cases where we don't have ST data, or at least, in the long-form scenario

# 1. we prepare a training set of ASR + ASR-turns + ST
mkdir -p ${data_output}/special-train-30s-only-asr-turn 
jq -s 'add' ${data_output}/fisher-callhome-train/data-asr-st.json \
    ${data_output}/fisher-callhome-train-30s/data-turns-asr.json \
    > ${data_output}/special-train-30s-only-asr-turn/data-turns-asr-t-st.json

# 2. we prepare a training set of ASR + ST + ST-turns
mkdir -p ${data_output}/special-train-30s-only-st-turn
jq -s 'add' ${data_output}/fisher-callhome-train/data-asr-st.json \
    ${data_output}/fisher-callhome-train-30s/data-turns-st.json \
    > ${data_output}/special-train-30s-only-st-turn/data-turns-asr-st-t.json

################################################################################
echo Done preparing the train, dev and test datasets FISHER-CALLHOME
exit 0