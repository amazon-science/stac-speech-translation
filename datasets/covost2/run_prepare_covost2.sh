#!/bin/bash

# Script to train prepare the commonvoice datasets for the multi-task ASR+ST

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM

##############################
echo "prepare COVOST2 data from Internet"
##############################
echo "This script needs commonvoice, enter and change the path if needed"
common_voice="/folder/to/datasets/common_voice_13/cv-corpus-13.0-2023-03-09"
covost_data="/folder/to/datasets/covost/splits"
covost_url="https://dl.fbaipublicfiles.com/covost"

# new language pairs to prepare
language_pairs_from_en="en_de"
language_pairs_to_en="de_en es_en fr_en"
language_pairs="$language_pairs_from_en $language_pairs_to_en"

for lang_pair in $(echo $language_pairs); do
    # Get the utterances IDs and filter out the recordings!
    _file="covost_v2.${lang_pair}.tsv.tar.gz"
    if ! [ -f "$covost_data/${_file}" ]; then
        echo "downloading language pair: $lang_pair"
        echo "$lang_pair not present... donwloading it"
        url=$(echo $covost_url/${_file} | tr -d '\"')
        wget -c --no-check-certificate "$url" -P $covost_data/
    fi
    
    if ! [ -d "$covost_data/covost_v2.${lang_pair}/covost_v2.${lang_pair}.dev.tsv" ]; then
        echo "extracting ${lang_pair}..."
        mkdir -p $covost_data/covost_v2.${lang_pair}
        tar -C $covost_data/covost_v2.${lang_pair} \
            -xvzf $covost_data/${_file}
    fi

done

##############################
echo "running the CoVost2 preparation script. Also, we convert to JSON files!"
##############################
[ -f $covost_data/.error ] && rm $covost_data/.error
for lang_pair in $(echo $language_pairs); do
    (
        source_lang=$(echo $lang_pair | cut -d'_' -f1)
        target_lang=$(echo $lang_pair | cut -d'_' -f2)
        _folder=$covost_data/covost_v2.${lang_pair}

        if ! [ -f "$_folder/covost_v2.${lang_pair}.dev.tsv" ]; then
            python get_covost_splits.py --version 2 \
            --src-lang $source_lang --tgt-lang $target_lang \
            --root $_folder \
            --cv-tsv ${common_voice}/${source_lang}/validated.tsv
        fi
        if ! [ -f "$_folder/covost_v2.${lang_pair}.dev.json" ]; then
            for set in $(echo dev test train); do
                python convert_covost_splits_to_json.py \
                    --input-json ${common_voice}/../data/${source_lang}/all.json \
                    --tsv-file $_folder/covost_v2.${lang_pair}.${set}.tsv \
                    --target-lang $target_lang
            done
        fi
    ) || touch $covost_data/.error &
done
wait
[ -f $covost_data/.error ] && echo "$0: there was a problem while preparing JSON fiels" && exit 1

##############################
echo "Printing some STATS"
##############################

for i in $(find $covost_data/ -mindepth 2 -iname "*.json"); do
    duration=$(grep '"duration"' $i | cut -d: -f2 |\
        cut -d, -f1 | awk '{x+=$1}END{print x/3600}')
    nb_lines=$(grep '"duration"' $i | wc -l | cut -d' ' -f1)
    echo "$i: $nb_lines | $duration"
done > $covost_data/stats.txt

echo "Done preparing the train, dev and test datasets FOR CommonVoice locales: $language_pairs"
exit 0