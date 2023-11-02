#!/usr/bin/env/python3
""" 
Script to perform inference with Whisper on datasets (in JSON format). 
We evaluate Fisher-CALLHOME test/dev subsets and CoVoST2.

Author
------
 * Juan Zuluaga-Gomez, 2023

"""

import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.utils.signal import Binarize, Peak
from tqdm import tqdm

FISHER_DATA_FOLDER = "/folder/to/datasets/fisher_callhome_spanish/data_processed/data"
# Global vars from PyAnnote
BATCH_AXIS = 0
TIME_AXIS = 1
SPEAKER_AXIS = 2


def store_rttm_file(rttm_output: List[str], rttm_output_file_path: str):
    """

    Parameters
    ----------
    rttm_output
    rttm_output_file_path

    Returns
    -------

    """
    with open(rttm_output_file_path, "w") as f:
        for line in rttm_output:
            f.write(line + "\n")


def main(args):
    """main function, see parse_arguments"""

    # get CLI input
    output_rttm = os.path.join(args.output_rttm)
    dataset_name = args.input_json_file.split("/")[-2]

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reading the ground evaluation data, which follows the convention below:
    print("reading the GT data to evaluate speaker change detection")
    with open(f"{args.input_json_file}", "r") as f:
        gt_data_raw = json.load(f)

    print(f"Using {args.pipeline} Pipeline.")
    if args.pipeline == "pyannote/speaker-diarization@2.1":
        # load the model and instantiate the inference object
        pipeline = Pipeline.from_pretrained(
            args.model_name,
            use_auth_token=args.auth_token,
        )
        pipeline = pipeline.to(device)

        # list with outputs
        hyp_rttm = []
        for key, values in tqdm(
            gt_data_raw.items(), desc=f"pre-processing [{dataset_name}]"
        ):
            # set path to wav file
            wav_path = values["wav"].replace("{data_root}", FISHER_DATA_FOLDER)

            # run diarization
            diarization = pipeline(wav_path, min_speakers=1, num_speakers=2)

            # append the outputs!
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # speaker speaks between turn.start and turn.end
                hyp_rttm.append(
                    f"SPEAKER {key} 1 {turn.start:.4f} {turn.duration:.4f} <NA> <NA> {speaker} <NA> <NA>"
                )

    else:
        model = Model.from_pretrained(
            args.model_name,
            use_auth_token=args.auth_token,
        )
        model = model.to(device)
        # VAD and SCD configs
        to_vad = lambda o: np.max(o, axis=SPEAKER_AXIS, keepdims=True)
        vad = Inference(model, pre_aggregation_hook=to_vad)

        to_scd = lambda probability: np.max(
            np.abs(np.diff(probability, n=1, axis=TIME_AXIS)),
            axis=SPEAKER_AXIS,
            keepdims=True,
        )
        scd = Inference(model, pre_aggregation_hook=to_scd)

        peak = Peak(alpha=0.05)
        binarize = Binarize(onset=0.5)

        # list with outputs
        hyp_rttm = []
        for key, values in tqdm(
            gt_data_raw.items(), desc=f"pre-processing [{dataset_name}]"
        ):
            # set path to wav file
            wav_path = values["wav"].replace("{data_root}", FISHER_DATA_FOLDER)

            # run VAD and binarize
            vad_prob = vad(wav_path)
            speech = binarize(vad_prob)

            # run SCD
            scd_prob = scd(wav_path)
            # Using a combination of Peak utility class (to detect peaks in the SCD output)
            detected_peaks = peak(scd_prob).crop(speech.get_timeline())

            for turn in detected_peaks:
                # speaker speaks between turn.start and turn.end
                hyp_rttm.append(
                    f"SPEAKER {key} 1 {turn.start:.4f} {turn.duration:.4f} <NA> <NA> SPK1 <NA> <NA>"
                )

    store_rttm_file(hyp_rttm, output_rttm)
    print(f"Finished running Pyannote and printed results in: {output_rttm}")

    return None


def parse_arguments():
    """function to parse input arguments from command-line"""
    parser = argparse.ArgumentParser(
        prog="script to run Pyannote on JSON dataset",
        description="perform speaker change detection with Pyannote",
    )

    parser.add_argument(
        "--auth_token",
        "-tok",
        required=True,
        type=str,
        help="authentication token from HuggingFace",
    )

    parser.add_argument(
        "--input_json_file",
        "-i",
        required=True,
        type=str,
        help="input JSON file with the data to be evaluated",
    )
    parser.add_argument(
        "--output_rttm",
        "-o",
        required=True,
        type=str,
        help="Output RTTM where to store all the resulting RTTM",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        default="pyannote/speaker-segmentation",
        # choices=["pyannote/speaker-segmentation"],
        help="Select the model name of (Pyannote) you want to use",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        type=str,
        default="pyannote/segmentation",
        choices=["pyannote/segmentation", "pyannote/speaker-diarization@2.1"],
        help="Select the segmentation approach: Pyannote - SCF or diarization",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
