"""
File to create individual wav files after acoustic segmentation with SHAS system (VAD)

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


def main():
    args = sys.argv
    segmentation_file = args[1]
    base_folder = args[2]
    data_folder = args[3]
    output_folder = args[4]

    # segmentation_file = os.path.join(base_folder, "shas_output.yaml")
    ground_truth_data = os.path.join(base_folder, "data.json")

    # Read JSON file: GROUND TRUH ANNOTATION
    with open(ground_truth_data, "r") as f:
        dataset_gt = json.load(f)

    # we need to collect the start and end in annotations for each file.
    # that way we can do a proper evaluation
    start_end_dict = {}
    for key, value in dataset_gt.items():
        _id = key.split("-")[0]
        # append the start to the first object
        if not _id in start_end_dict:
            start_end_dict[_id] = {
                "start": float(key.split("-")[2]),
                "end": float(key.split("-")[3]),
            }
        start_end_dict[_id]["end"] = float(key.split("-")[3])

    # Read YAML file: VAD OUTPUT
    with open(segmentation_file, "r") as stream:
        segmented_data = yaml.safe_load(stream)

    # split the wav files and print a new JSON file for evaluation!
    output_json_file_asr, output_json_file_st = {}, {}
    # for segmented in segmented_data:
    for segmented in tqdm(segmented_data, desc=f"pre-processing [{ground_truth_data}]"):
        # get the metadata of the VAD output
        _id = segmented["wav"].split(".")[0]
        start = int(float(segmented["offset"]) * 100)
        duration = int(float(segmented["duration"]) * 100)
        end = start + duration

        # get the min-max start-end of the ground truth segmentation
        min_start_allowed = start_end_dict[_id]["start"]
        max_end_allowed = start_end_dict[_id]["end"]

        utterance_id = f"{_id}-{0}-{start:06d}-{end:06d}"

        # that means that the given VAD segment is outside of the allowed boundaries
        # in the whole file, this might happen at the begining of files or at the end
        if (start < min_start_allowed and end < min_start_allowed) or (
            start > max_end_allowed and end > max_end_allowed
        ):
            print(f"error processing this file {utterance_id}")
            continue

        wav_path = os.path.join(data_folder, segmented["wav"])
        wav_save_path = os.path.join(
            os.path.abspath(output_folder), utterance_id + ".wav"
        )

        if not os.path.exists(f"{wav_save_path}"):
            segment_audio(
                audio_path=wav_path, start=start, end=end, save_path=wav_save_path
            )

        for target_lang, task, output_json_file in zip(
            ["es", "en"],
            ["transcription", "translation"],
            [output_json_file_asr, output_json_file_st],
        ):
            output_json_file[utterance_id] = {
                "wav": wav_save_path,
                "source_lang": "es",
                "target_lang": target_lang,
                "segments_start": 0,
                "segments_duration": f"{duration/100:.2f}",
                "segments_channel": "0",
                "duration": f"{duration/100:.2f}",
                "task": task,
                "transcription": "",
                "translation_0": "",
            }
            # append the new JSON output
            output_json_file.update(output_json_file)

    # print both objects!
    for task in ["asr", "st"]:
        output_file = os.path.join(base_folder, f"data-resegmented-{task}.json")
        output_json_file = (
            output_json_file_asr if task == "asr" else output_json_file_st
        )

        print(f"printing new JSON object in: {output_file}")
        with open(output_file, mode="w", encoding="utf-8") as json_f:
            json.dump(output_json_file, json_f, indent=2, ensure_ascii=False)


def segment_audio(
    audio_path: str,
    start: int,
    end: int,
    save_path: str,
    sample_rate: int = 16000,
):
    """segment and resample audio"""

    start = int(start / 100 * 16000)
    end = int(end / 100 * 16000)
    num_frames = end - start

    data, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)
    data = torch.unsqueeze(data[0], 0)
    torchaudio.save(save_path, src=data, sample_rate=sample_rate)


if __name__ == "__main__":
    main()
