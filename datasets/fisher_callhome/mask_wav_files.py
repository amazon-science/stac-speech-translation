"""
File to mask WAVs files with ground truth segmentation. 
We basically set to 0 part of the signal where there's no annotation in the ground truth
FISHER-CALLHOME dataset

Author
------
 * Juan Zuluaga-Gomez, 2023
"""

import json
import os
import sys

import torch
import torchaudio
import yaml
from tqdm import tqdm

SAMPLERATE = 16000


def main():
    """function to mask the wav files of fisher-callhome with 0s"""

    args = sys.argv
    ground_truth_data = args[1]
    input_folder = args[2]
    output_folder = args[3]

    segmentation_file = os.path.join(input_folder, "shas_output.yaml")
    # ground_truth_data = os.path.join(input_folder)

    # Read JSON file: GROUND TRUH ANNOTATION
    with open(ground_truth_data, "r") as f:
        dataset_gt = json.load(f)

    # we need to collect the start and end in annotations for each file.
    # that way we can do a proper evaluation
    start_end_dict = {}
    for key, value in dataset_gt.items():
        _id = key.split("-")[0]

        # append the start and end of frames of each sample
        start_frame = int(key.split("-")[2])
        start_frame = int((start_frame / 100) * 16000)
        end_frame = float(key.split("-")[3])
        end_frame = int((end_frame / 100) * 16000)

        if not _id in start_end_dict:
            start_end_dict[_id] = [[start_frame, end_frame]]
        start_end_dict[_id].append([start_frame, end_frame])

    # for segmented in segmented_data:
    for utt_id in tqdm(start_end_dict, desc=f"pre-processing [{ground_truth_data}]"):
        wav_path = os.path.join(input_folder, f"{utt_id}.wav")
        wav_save_path = os.path.join(output_folder, f"{utt_id}.wav")

        wav_data, _ = torchaudio.load(wav_path)

        # create a mask where speech activity == 1 else, silence==0
        mask = torch.zeros(wav_data.shape[1])
        for sample in start_end_dict[utt_id]:
            mask[sample[0] : sample[1]] = 1

        # convert mask back to 1 x D and apply mask
        mask = mask[None, :]
        masked_output = wav_data * mask.int().float()

        masked_output = torch.unsqueeze(masked_output[0], 0)
        torchaudio.save(
            wav_save_path,
            src=masked_output,
            encoding="PCM_S",
            bits_per_sample=16,
            sample_rate=SAMPLERATE,
        )


if __name__ == "__main__":
    main()
