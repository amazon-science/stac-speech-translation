#!/bin/bash

# Script to train prepare the commonvoice datasets for the multi-task ASR+ST

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM

data_folder="/folder/to/datasets/common_voice_13/cv-corpus-13.0-2023-03-09"
save_folder="/folder/to/datasets/common_voice_13/data"

##############################
echo "get CommonVoice data from Internet"
##############################
bash get_data.sh $data_folder

languages="es en de fr"

##############################
echo "preparing the dev/test/train/validated TSV files"
##############################
for locale in $(echo $languages); do    
    # validated set contains dev and test, we need to filter out those ones
    for set in $(echo "dev test"); do
        cut -d$'\t' -f2 ${data_folder}/$locale/$set.tsv | grep "mp3" > \
            ${data_folder}/$locale/$set.ids
    done 
    
    # now filter the ids DEV and TEST from validated, so we have a clean train set
    grep -vFwf ${data_folder}/$locale/dev.ids \
        ${data_folder}/$locale/validated.tsv |\
        grep -vFwf ${data_folder}/$locale/test.ids >\
        ${data_folder}/$locale/train_validated.tsv
done

##############################
echo "preparing JSON files for all languages"
##############################
[ -f $save_folder/.error ] && rm $save_folder/.error
for locale in $(echo $languages); do
    (
        python3 prepare_cv.py \
            --language $locale \
            --data-folder $data_folder \
            --save-folder $save_folder \
            --threshold 45
        
        echo "joining all JSON files in one file: $save_folder/$locale"
        jq -s 'add' ${save_folder}/${locale}/{dev,test,train_validated}.json \
            > ${save_folder}/${locale}/all.json
    ) || touch $save_folder/.error &
done
wait
[ -f $save_folder/.error ] && echo "$0: there was a problem while aligning the files" && exit 1

##############################
echo "Printing some STATS"
##############################

for i in $(find $save_folder/ -mindepth 2 -iname "*.json"); do
    duration=$(grep '"duration"' $i | cut -d: -f2 |\
        cut -d, -f1 | awk '{x+=$1}END{print x/3600}')
    nb_lines=$(grep '"duration"' $i | wc -l | cut -d' ' -f1)
    echo "$i: $nb_lines | $duration"
done > $save_folder/stats.txt


echo "Done preparing the train, dev and test datasets FOR CommonVoice locales: $languages"
exit 0