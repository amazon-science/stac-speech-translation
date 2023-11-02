#!/bin/bash

# Script to prepare the MSLT dataset (see Github: https://github.com/MicrosoftTranslator/MSLT-Corpus)
# You can donwload it from: 
# Version 1: https://www.microsoft.com/en-us/download/details.aspx?id=54689
# Version 1.1: https://www.microsoft.com/en-us/download/details.aspx?id=55951

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM

##############################
echo "preparing JSON files for MSLT dataset"
##############################
data_folder=/folder/to/datasets/mslt
save_folder=/folder/to/datasets/mslt/data
versions="1 1_1"
accented_letters="True"
max_duration=45

[ -f $save_folder/.error ] && rm $save_folder/.error
for version in $(echo $versions); do
    (
        python3 mslt_prepare.py \
            --data-folder ${data_folder}/v${version} \
            --save-folder $save_folder \
            --version $version \
            --accented-letters $accented_letters \
            --duration-threshold $max_duration
    ) || touch $save_folder/.error &
done
wait
[ -f $save_folder/.error ] && echo "$0: there was a problem while preparing JSON files" && exit 1

##############################
echo "Printing some STATS"
##############################

for i in $(find $save_folder/ -mindepth 2 -iname "*.json"); do
    duration=$(grep '"duration"' $i | cut -d: -f2 |\
        cut -d, -f1 | awk '{x+=$1}END{print x/3600}')
    echo "$i: $duration"
done > $save_folder/stats.txt

echo "Done preparing the DEV/TEST sets FOR MSLT DATASET, versions: $versions"
exit 0