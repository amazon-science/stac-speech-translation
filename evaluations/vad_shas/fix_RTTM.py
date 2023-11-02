#!/usr/bin/env/python3
""" 
Script to fix RTTM file produced by model!. 

Author
------
 * Juan Zuluaga-Gomez, 2023
"""

import argparse
import csv
import json
import os
from typing import List


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


import ipdb


def main(args):
    """main function, see parse_arguments"""
    _NB_errors = 0

    # get CLI input
    gt_json = os.path.join(args.gt_json)
    pred_rttm = os.path.join(args.pred_rttm)
    output_folder = os.path.join(args.output_folder)

    # reading the ground data, with start and duration per sample
    print("reading the GT ")
    with open(f"{gt_json}", "r") as f:
        gt_data_raw = json.load(f)

    # load the RTTM outputs from the model
    pred_data_raw = {}
    with open(f"{pred_rttm}", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            line = line[0].strip().split()
            rec_id = line[1].replace("-st", "").replace("-asr", "")
            time, dur = [line[3], line[4]]
            speaker_id = line[7]

            # get the start time to set relative time instead to 'absolute'
            start_time = int(line[1].split("-")[2]) / 100

            # update the start time
            abs_time = float(time) - start_time
            abs_time = abs_time if abs_time > 0 else 0
            time = f"{abs_time:.4f}"

            # append in dictionary the objects
            if rec_id not in pred_data_raw:
                pred_data_raw[rec_id] = [[time, dur, speaker_id]]
            else:
                pred_data_raw[rec_id].append([time, dur, speaker_id])

    # this is done to always increase the time after new sample is appended
    trailing_time = 0
    # list to save the RTTMs
    ref_rttm, hyp_rttm = [], []

    for sample in gt_data_raw:
        sample = gt_data_raw[sample]
        sample_start = [float(i) for i in sample["segments_start"].split(" ")]
        sample_duration = [float(i) for i in sample["segments_duration"].split(" ")]
        utt_id = sample["wav"].split("/")[-1].replace(".wav", "")
        # ipdb.set_trace()
        # check that the sample is also produced by the model
        if utt_id not in pred_data_raw:
            _NB_errors += 1
            continue

        # get the RTTM for ground truth
        for start, duration in zip(sample_start, sample_duration):
            # construct the Grount truth RTTM file!
            start = trailing_time + start
            ref_rttm.append(
                f"SPEAKER {utt_id} 1 {start:.2f} {duration} <NA> <NA> SPK1 <NA> <NA>"
            )

        # get the RTTM for prediction
        for start, duration, speaker_id in pred_data_raw[utt_id]:
            start, duration = float(start), float(duration)
            start = trailing_time + start
            hyp_rttm.append(
                f"SPEAKER {utt_id} 1 {start:.2f} {duration} <NA> <NA> {speaker_id} <NA> <NA>"
            )

        # update the trailing time, 5 seconds of space between utterances
        # ipdb.set_trace()
        end_gt = trailing_time + float(sample["duration"])
        trailing_time = int(end_gt + 5)
    # ipdb.set_trace()
    # set the output paths and save the RTTM files
    file_id = pred_rttm.split("/")[-1].split(".csv")[0]

    ref_rttm_file_path = os.path.join(output_folder, f"{file_id}.ref.rttm")
    hyp_rttm_file_path = os.path.join(output_folder, f"{file_id}.hyp.rttm")

    store_rttm_file(ref_rttm, ref_rttm_file_path)
    store_rttm_file(hyp_rttm, hyp_rttm_file_path)

    return None


def parse_arguments():
    """function to parse input arguments from command-line"""
    parser = argparse.ArgumentParser(
        prog="script to re-align long-form ST files with mwerSegmenter",
        description="realign long-form text with mwerSegmenter",
    )

    parser.add_argument(
        "--gt_json",
        "-gt",
        required=True,
        type=str,
        help="input JSON file with the data to be evaluated",
    )
    parser.add_argument(
        "--pred_rttm",
        "-pred",
        required=True,
        type=str,
        help="Prediction RTTM file to fix",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        required=True,
        type=str,
        help="Output folder where to store the new RTTM files (GT and PRED)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
