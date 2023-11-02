#!/bin/bash

# Script to run SHAS tool to segment long-form conversational audio in small recordings!
# Check the scripts in: https://github.com/mt-upc/SHAS

# kill subprocess on exit
trap "pkill -P $$" EXIT SIGINT SIGTERM
trap "echo Exited!; exit;" SIGINT SIGTERM

# we will the segmentation file for SPANISH --> Fisher-callhome dataset
spanish_model="https://drive.google.com/uc?export=download&confirm=BlwG&id=1f73JKIv9Z7YarIHNhxABm8H3H4dXygwC"
shas_repo="https://github.com/mt-upc/SHAS"

# DATA_FOLDER
fisher_data=/folder/to/datasets/fisher_callhome_spanish/LDC2010T04/fisher_spa/data/speech
callhome_data=/folder/to/datasets/fisher_callhome_spanish/LDC96T17/ch_sp/callhome/spanish/speech

data_prepared="data/"
repo_dir=audio_segmenter
mkdir -p $repo_dir


##############################
echo "Stage 1: cloning GitHub repo"
##############################

if ! [ -d $repo_dir/shas ]; then
    echo "cloning repo $shas_repo in $repo_dir"
    git clone $shas_repo $repo_dir/shas
else
    echo "repository SHAS ($shas_repo) already present"
fi

##############################
echo "Stage 2: creating conda environment and downloading spanish model (SHAS)"
##############################

# create conda environment and run activate it
# set to TRUE IF YOU WANT TO CREATE AGAIN AN ENVIRONMENT
if false; then
    conda env create -f ${repo_dir}/shas/environment.yml && \
    conda activate shas
fi

# donwload the file 
if ! [ -f $repo_dir/es_sfc_model_epoch-2.pt ]; then
    echo "donwloading Spanish Checkpoint in $repo_dir"
    wget "$spanish_model" -O $repo_dir/es_sfc_model_epoch-2.pt
else
    echo "Spanish CKPT already present: $repo_dir/es_sfc_model_epoch-2.pt"
fi

if ! [ -d $repo_dir/data/callhome-evltest-webrct ]; then

    # Make the output with WEBRCT
    for subset in $(echo "dev dev2 test"); do
        output_folder=$repo_dir/data/${subset}-webrct
        mkdir -p $output_folder/wavs

        grep ': {'  $data_prepared/$subset/data-st.json | \
            cut -d'"' -f2 | cut -d- -f1 | sort | uniq > $output_folder/wavs/ids

        # reading file by file
        while read -r line
        do
            file_path="$fisher_data/${line}.sph"
            cp $file_path $output_folder/wavs/
        done < $output_folder/wavs/ids
        
        ls ${output_folder}/wavs/*.* |\
            parallel -j 4 ffmpeg -i {} -ac 1 -ar 16000 -c:a pcm_s16le -hide_banner -loglevel error {.}.wav
        # copy the file with annotations!
        cp $data_prepared/$subset/data-st.json $output_folder/data.json
    done

    # copy ONLY CALLHOME datasets
    for subset in $(echo "callhome-devtest callhome-evltest"); do
        output_folder=$repo_dir/data/${subset}-webrct
        mkdir -p $output_folder/wavs

        file_ids=$(grep ': {'  $data_prepared/$subset/data-st.json | \
            cut -d'"' -f2 | cut -d- -f1 | sort | uniq)
        
        for line in $(echo $file_ids | tr "\n" " "); do
            # get the input file_path
            file_path=$(echo $callhome_data/$(echo $subset | cut -d- -f2)/${line}.wav)
            ffmpeg -i ${file_path} -ac 1 -ar 16000 -c:a pcm_s16le -hide_banner -loglevel error $output_folder/wavs/${line}.wav
        done
        # copy the file with annotations!
        cp $data_prepared/$subset/data-st.json $output_folder/data.json

    done

    for subset in $(echo "dev dev2 test callhome-devtest callhome-evltest"); do
        echo "Performing VAD with WEBRCT algorithm on $subset"

        output_folder=$(realpath $repo_dir/data/${subset}-webrct)
        mkdir -p $output_folder/{masked_wavs,resegmented}

        # JSON file with dataset + ground truth annotations, input and output folder
        if ! [ -f $output_folder/masked_wavs/.done* ]; then
            ##############################
            echo "Key step: Masking the input WAV signal with 0s with ground truth segmentation"
            ##############################
            python3 mask_wav_files.py \
                $output_folder/data.json \
                $output_folder/wavs \
                $output_folder/masked_wavs
            touch $output_folder/masked_wavs/.done
        fi

        if ! [ -f $output_folder/webrct_output.yaml ]; then
            # Pause-based segmentation with webrtc VAD:
            frame_length=10
            aggressiveness_mode=1
            CUDA_VISIBLE_DEVICES=0 \
                python ${repo_dir}/shas/src/segmentation_methods/pause_based.py \
                -wavs $output_folder/masked_wavs \
                -yaml $output_folder/webrct_output.yaml \
                -l=$frame_length \
                -a=$aggressiveness_mode

            python3 create_json_and_segment.py \
                $output_folder/webrct_output.yaml \
                $output_folder/ \
                $output_folder/masked_wavs \
                $output_folder/resegmented
        fi
    done
fi
exit 1
##############################
echo "Doing VAD segmentation with SHAS - VAD system "
##############################

# select the min-max ranges that we want to create our files!
max_min_list="10_15 15_20 25_30 1_30 1_10 1_15 1_20 1_25 5_20 5_30 10_30"
# max_min_list="1_10 1_15 1_20 1_25"
# max_min_list="5_20 5_30 10_30"
max_min_list=($max_min_list)

for max_min_pair in "${max_min_list[@]}"; do
    # (
        dac_min_segment_length=$(echo $max_min_pair | cut -d'_' -f1)
        dac_max_segment_length=$(echo $max_min_pair | cut -d'_' -f2)

        ##############################
        echo "Stage 3: Copying and pre-processing the WAV files of Fisher-Callhome"
        ##############################

        # copy fisher-callhome datasets
        if ! [ -d $repo_dir/data/dev-${dac_min_segment_length}-to-${dac_max_segment_length}/wavs ]; then
            for subset in $(echo "dev dev2 test"); do
                output_folder=$repo_dir/data/${subset}-${dac_min_segment_length}-to-${dac_max_segment_length}
                mkdir -p $output_folder/wavs

                grep ': {'  $data_prepared/$subset/data-st.json | \
                    cut -d'"' -f2 | cut -d- -f1 | sort | uniq > $output_folder/wavs/ids

                # reading file by file
                while read -r line
                do
                    file_path="$fisher_data/${line}.sph"
                    cp $file_path $output_folder/wavs/
                done < $output_folder/wavs/ids
                
                ls ${output_folder}/wavs/*.* |\
                    parallel -j 4 ffmpeg -i {} -ac 1 -ar 16000 -c:a pcm_s16le -hide_banner -loglevel error {.}.wav
                # copy the file with annotations!
                cp $data_prepared/$subset/data-st.json $output_folder/data.json
            done
        fi

        if ! [ -d $repo_dir/data/callhome-devtest-${dac_min_segment_length}-to-${dac_max_segment_length}/wavs ]; then
            # copy ONLY CALLHOME datasets
            for subset in $(echo "callhome-devtest callhome-evltest"); do
                output_folder=$repo_dir/data/${subset}-${dac_min_segment_length}-to-${dac_max_segment_length}
                mkdir -p $output_folder/wavs

                file_ids=$(grep ': {'  $data_prepared/$subset/data-st.json | \
                    cut -d'"' -f2 | cut -d- -f1 | sort | uniq)
                
                for line in $(echo $file_ids | tr "\n" " "); do
                    # get the input file_path
                    file_path=$(echo $callhome_data/$(echo $subset | cut -d- -f2)/${line}.wav)
                    ffmpeg -i ${file_path} -ac 1 -ar 16000 -c:a pcm_s16le -hide_banner -loglevel error $output_folder/wavs/${line}.wav
                done
                # copy the file with annotations!
                cp $data_prepared/$subset/data-st.json $output_folder/data.json

            done
        fi

        ##############################
        echo "Stage 5: Running the segmentation algorithm"
        ##############################

        for subset in $(echo "dev dev2 test callhome-devtest callhome-evltest"); do
            echo "Performing VAD with SHAS algorithm on $subset"

            output_folder=$(realpath $repo_dir/data/${subset}-${dac_min_segment_length}-to-${dac_max_segment_length})
            mkdir -p $output_folder/{masked_wavs,resegmented}

            # JSON file with dataset + ground truth annotations, input and output folder
            if ! [ -f $output_folder/masked_wavs/.done* ]; then
                ##############################
                echo "Key step: Masking the input WAV signal with 0s with ground truth segmentation"
                ##############################
                python3 mask_wav_files.py \
                    $output_folder/data.json \
                    $output_folder/wavs \
                    $output_folder/masked_wavs
                touch $output_folder/masked_wavs/.done
            fi

            # processing file by file!
            if ! [ -f $output_folder/shas_output.yaml ]; then
                CUDA_VISIBLE_DEVICES=3 python ${repo_dir}/shas/src/supervised_hybrid/segment.py \
                    --inference_batch_size=24 \
                    --path_to_wavs $output_folder/masked_wavs \
                    --path_to_checkpoint $repo_dir/es_sfc_model_epoch-2.pt \
                    --path_to_segmentation_yaml $output_folder/shas_output.yaml \
                    --dac_min_segment_length=$dac_min_segment_length \
                    --dac_max_segment_length=$dac_max_segment_length
            
                python3 create_json_and_segment.py \
                    $output_folder/shas_output.yaml \
                    $output_folder/ \
                    $output_folder/masked_wavs \
                    $output_folder/resegmented
            fi


        done
    # ) &
done
wait

################################################################################
echo "Done segmentation audio files"
exit 0