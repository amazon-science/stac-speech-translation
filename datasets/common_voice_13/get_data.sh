#!/usr/bin/env bash
#
# Prepare CommonVoice datasets

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <download_dir>"
  echo "e.g.: $0 /tmp/commonvoice_raw/"
  exit 1
fi

# variables
download_dir=$(realpath $1)
echo $download_dir

echo "Data Download"
common_voice_url="https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-13.0-2023-03-09/cv-corpus-13.0-2023-03-09"

# the four locales for the experiments with Fisher-CALLHOME and MSLT
languages="es en de fr"

for lang in $(echo $languages); do
    url=${common_voice_url}-${lang}.tar.gz

    if ! [ -f "$download_dir/cv-corpus-13.0-2023-03-09-${lang}.tar.gz" ]; then
        echo "${lang}: locale not present... donwloading it in $download_dir"
        wget -c -P $download_dir/ $url
    else
        echo "${lang}: locale present... skip"
    fi

    if ! [ -d "$download_dir/cv-corpus-13.0-2023-03-09/${lang}" ]; then
        echo "${lang}: extracting dataset"
        tar -C $download_dir/ -xvzf $download_dir/cv-corpus-13.0-2023-03-09-${lang}.tar.gz
    else
        echo "${lang}: folder already present in $download_dir"
    fi
done

echo "done downloading and extractig the CommonVoice corpora"
exit 1
