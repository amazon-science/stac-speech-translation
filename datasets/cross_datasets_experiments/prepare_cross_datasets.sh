#!/bin/bash

# Script to prepare the cross datasets experiments, check the folder of each used dataset 
# below if you want more information on how to use or prepare them

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM

##############################
echo "preparing cross experiments"
##############################
datasets_folder=/folder/to/datasets/
mslt_dataset=${datasets_folder}/mslt/data/v_1/
commonvoice=${datasets_folder}/common_voice_13/data/
covost=${datasets_folder}/covost/splits/
fisher_callhome=${datasets_folder}/fisher_callhome_spanish/data_processed/data

##############################
echo "***************************"
echo "Preparing experiments for MSLT dataset and Fisher-callhome dataset"
echo "We will prepare data in 3 directions (based on Covost2 availability)"
echo "Exp-1: source-->target: EN --> EN/DE"
echo "Exp-2: source-->target: DE --> EN/DE"
echo "Exp-3: source-->target: FR --> EN/FR"
echo "For Fisher-callhome dataset"
echo "Exp-4: source-->target: ES --> EN/ES"
echo "Exp-5: source-->target: ALL <--> ALL"
echo "***************************"
##############################

##############################
echo "Stage 1: prepare the data for MSLT + covost2 datasets"
##############################
output_folder=data_covost_mslt/
mkdir -p $output_folder
################################################################################
echo "we will merge, CommonVoice and Covost2 data in: $subfolder"

language_pairs_from_en="en_de en_ca en_sl en_et en_sv-SE"
language_pairs_to_en="de_en fr_en es_en ca_en it_en pt_en et_en nl_en lv_en sl_en cy_en"
locale_pairs="ca_en"

[ -f $output_folder/.error ] && rm $output_folder/.error
for locale_pair in $(echo $locale_pairs); do
    (
        # get the locales pair
        source_locale=$(echo $locale_pair | cut -d'_' -f1)
        target_locale=$(echo $locale_pair | cut -d'_' -f2)

        echo "Exp: source-->target: $source_locale --> ${source_locale}/${target_locale}"
        echo "***************************"

        join=${source_locale}_${target_locale}
        subfolder=$output_folder/${source_locale}_to_${source_locale}.and.${target_locale}
        mkdir -p $subfolder

        # copy the commonvoice and covost original train sets:
        cp ${commonvoice}/$source_locale/train.json $subfolder/train_cv_${source_locale}.json
        cp ${covost}/covost_v2.$join/covost_v2.${join}.train.json \
            $subfolder/train_covost_v2.${join}.train.json            
        
        # now process the dev/test of commonvoice
        python3 join_json.py ${commonvoice}/$source_locale/dev.json $subfolder/cv_${source_locale}_dev.json
        python3 join_json.py ${commonvoice}/$source_locale/test.json $subfolder/cv_${source_locale}_test.json

        # now process the dev/test of COVOST2
        python3 join_json.py ${covost}/covost_v2.${join}/covost_v2.${join}.dev.json \
            $subfolder/covost_${join}_dev.json
        python3 join_json.py ${covost}/covost_v2.${join}/covost_v2.${join}.test.json \
            $subfolder/covost_${join}_test.json

        # preparing the MSLT datasets if en_de de_en fr_en
        if [[ "$locale_pair" == "en_de" ]] || \
            [[ "$locale_pair" == "de_en" ]] || \
            [[ "$locale_pair" == "fr_en" ]]; then

            for subset in $(echo "dev test"); do
                for locale in $(echo "$source_locale $target_locale"); do
                    python3 join_json.py \
                        ${mslt_dataset}/mslt_1v__${subset}_${source_locale}_${locale}.json \
                        $subfolder/mslt_1v__${subset}_${source_locale}_${locale}.json
                done
            done
        fi

        # new create a new version of each dataset, where 
        # each segments is around 30s long!
        for file in $(find $subfolder/ -maxdepth 1 -iname "*json"); do
            python3 concat_json.py --input-json $file --target-duration 30
        done

        # Join the original versions of train sets
        python3 join_json.py ${train_files} \
            $subfolder/train_cv_${source_locale}.json \
            $subfolder/train_covost_v2.${join}.train.json \
            $subfolder/train_cv_covost.${join}.json

        # now join the versions concatenated to 30s
        python3 join_json.py ${train_files} \
            $subfolder/train_cv_${source_locale}-30s.json \
            $subfolder/train_covost_v2.${join}.train-30s.json \
            $subfolder/train_cv_covost-30s.${join}.json
        
        # now join both!
        python3 join_json.py \
            $subfolder/train_cv_covost.${join}.json \
            $subfolder/train_cv_covost-30s.${join}.json \
            $subfolder/train-and-30s.${join}.json

        # now join the MSLT DEV datasets per language pair to fine-tune on it!
        # get all the dev files
        dev_to_train=$(ls $subfolder/mslt_1v__dev*)
        python3 join_json.py \
            $dev_to_train $subfolder/train_mslt_1v-and-30s.json

        echo "Finish preparing: source-->target: $source_locale --> ${source_locale}/${target_locale}"
        echo "***************************"
    ) || touch $output_folder/.error &
done
wait
[ -f $output_folder/.error ] && echo "$0: there was a problem while preparing JSON files" && exit 1

##############################
echo "Printing some STATS"
##############################

for i in $(find $output_folder/ -mindepth 2 -iname "*.json"); do
    duration=$(grep '"duration"' $i | cut -d: -f2 |\
        cut -d, -f1 | awk '{x+=$1}END{print x/3600}')
    nb_lines=$(grep '"duration"' $i | wc -l | cut -d' ' -f1)
    echo "$i: $nb_lines | $duration"
done > $output_folder/stats.txt


##############################
echo "Stage 2: prepare the data for fisher-callhome + covost2 datasets"
##############################
output_folder=data_covost_fisher/
mkdir -p $output_folder
################################################################################
echo "we will merge, CommonVoice and Covost2 data in: $subfolder"

locale_pair="es_en"

# get the locales pair
source_locale=$(echo $locale_pair | cut -d'_' -f1)
target_locale=$(echo $locale_pair | cut -d'_' -f2)

echo "Exp: source-->target: $source_locale --> ${source_locale}/${target_locale}"
echo "***************************"

join=${source_locale}_${target_locale}
subfolder=$output_folder/${source_locale}_to_${source_locale}.and.${target_locale}
mkdir -p $subfolder

# merging covost and commonvoice
# copy the commonvoice and covost original train sets:
cp ${commonvoice}/$source_locale/train.json $subfolder/train_cv_${source_locale}.json
cp ${covost}/covost_v2.$join/covost_v2.${join}.train.json \
    $subfolder/train_covost_v2.${join}.train.json            

# now process the dev/test of commonvoice
python3 join_json.py ${commonvoice}/$source_locale/dev.json $subfolder/cv_${source_locale}_dev.json
python3 join_json.py ${commonvoice}/$source_locale/test.json $subfolder/cv_${source_locale}_test.json

# now process the dev/test of COVOST2 
python3 join_json.py ${covost}/covost_v2.${join}/covost_v2.${join}.dev.json \
    $subfolder/covost_${join}_dev.json
python3 join_json.py ${covost}/covost_v2.${join}/covost_v2.${join}.test.json \
    $subfolder/covost_${join}_test.json

# Concatenate the files to create segments up to 30s long!
for file in $(find $subfolder/ -maxdepth 1 -iname "*json"); do
    python3 concat_json.py --input-json $file --target-duration 30
done

# Join the original versions
python3 join_json.py ${train_files} \
    $subfolder/train_cv_${source_locale}.json \
    $subfolder/train_covost_v2.${join}.train.json \
    $subfolder/train_cv_covost.${join}.json

# now join the versions concatenated up to 30s
python3 join_json.py ${train_files} \
    $subfolder/train_cv_${source_locale}-30s.json \
    $subfolder/train_covost_v2.${join}.train-30s.json \
    $subfolder/train_cv_covost-30s.${join}.json

# now join the original fisher-callhome dataset + the one concatenated
python3 join_json.py \
    ${fisher_callhome}/fisher-callhome-train-and-30s/data-turns-asr-st.json \
    $subfolder/train_cv_covost.${join}.json \
    $subfolder/train_cv_covost-30s.${join}.json \
    $subfolder/train-and-30s.${join}.json

# linking the Fisher-callhome dev/test sets into the experiment folder
for subset in $(echo "dev dev2 test callhome-evltest callhome-devtest"); do
    ln -sf ${fisher_callhome}/$subset $subfolder/
    ln -sf ${fisher_callhome}/${subset}-30s $subfolder/
done

echo "Finish preparing: source-->target: $source_locale --> ${source_locale}/${target_locale}"
echo "***************************"

##############################
echo "Printing some STATS"
##############################

for i in $(find $output_folder/ -mindepth 2 -iname "*.json"); do
    duration=$(grep '"duration"' $i | cut -d: -f2 |\
        cut -d, -f1 | awk '{x+=$1}END{print x/3600}')
    nb_lines=$(grep '"duration"' $i | wc -l | cut -d' ' -f1)
    echo "$i: $nb_lines | $duration"
done > $output_folder/stats.txt

##############################
echo "Stage 4: joining all languages (including new ones), CommonVoice+covost2+Fisher and evaluate on MSLT"
##############################
output_folder=data_covost_mslt/all2_to_all2
################################################################################
mkdir -p $output_folder

# get all the language pairs: 
# the *and* ensures to take CV and Covost. Non-english source because it is repeated.
# we added it below!
all_train_datasets=$(ls data_covost_mslt/*/train* |\
    grep -v train_cv_covost | grep "cv\|covost_v2" | grep -v "train_cv_en")

# now join all language pairs! we add only once CV english and FISHER dataset. es_en is already present
python3 join_json.py \
    data_covost_mslt/en_to_en.and.de/train_cv_en.json \
    data_covost_mslt/en_to_en.and.de/train_cv_en-30s.json \
    $all_train_datasets \
    ${fisher_callhome}/fisher-callhome-train-and-30s/data-turns-asr-st.json \
    $output_folder/train-and-30s.all2_all2.json


# now join all language pairs! we add only once CV english and FISHER dataset. es_en is already present
python3 join_json.py \
    data_covost_mslt/de_to_de.and.en/train_mslt_1v-and-30s.json \
    data_covost_mslt/en_to_en.and.de/train_mslt_1v-and-30s.json \
    data_covost_mslt/fr_to_fr.and.en/train_mslt_1v-and-30s.json \
    $output_folder/train_mslt_1v_all-and-30s.json

echo "Finish preparing: source-->target: $source_locale --> ${source_locale}/${target_locale}"
echo "***************************"

echo "Done merging JSON for cross datasets experiments"
exit 0