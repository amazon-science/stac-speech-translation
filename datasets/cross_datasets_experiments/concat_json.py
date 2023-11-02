"""
Take a JSON file and concat segments up to amount of seconds given

Author
------
 * Juan Zuluaga-Gomez, 2023

"""
import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        "-i",
        type=str,
        required=True,
        help="Input JSON file to concatenate",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=int,
        default=None,
        help="Output JSON file, otherwise, it will in the same folder as input",
    )
    parser.add_argument(
        "--target-duration",
        "-t",
        type=int,
        required=True,
        help="Target duration in seconds to concatenate new segments. \
                            Needs to be a int",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=8888, help="random seed for sampling."
    )

    return parser.parse_args()


def main():
    args = get_args()

    random.seed(args.seed)
    in_file = args.input_json
    target_duration = int(args.target_duration)

    # set output file based on input args
    output_file = (
        in_file.split(".json")[0] + f"-{target_duration}s.json"
        if args.output_json is None
        else args.output_json
    )

    print(f"concatenating segments in {in_file} up to {target_duration} seconds")
    print(f"output file is in: {output_file}")

    # load JSON file
    with open(in_file, "r") as f:
        dataset = json.load(f)

    # get a list with the keys of the whole dataset
    keys_list = [i for i in dataset.keys()]

    new_dataset = {}
    current_dur = 0
    current_obj = {}
    while len(dataset) > 0:
        # get random IDx, then get the key from key_list and then get the element
        # it's way faster than taking a random element from dictionaryis faster
        rand_idx = random.randrange(len(keys_list))
        key = keys_list[rand_idx]
        value = dataset[key]

        # create this field if not present
        if "transcription_and_translation" not in value:
            value[
                "transcription_and_translation"
            ] = f"{value['transcription']} \n {value['translation_0']}"

        # option 1: create the object if empty
        if len(current_obj) == 0 and len(dataset) > 0:
            value["segments_start"] = "0"
            value["segments_duration"] = f"{value['duration']:.2f}"
            value["segments_channel"] = f"0"
            current_obj = [key, value]
            del dataset[key]
            keys_list.pop(rand_idx)
            continue

        # option 2: object present and we can append the duration
        if (
            float(current_obj[1]["duration"]) + float(value["duration"])
            < target_duration
        ):
            # append current sample to current_object objects

            # append transcript and translation
            current_obj[1]["translation_0"] = (
                current_obj[1]["translation_0"] + " [turn] " + value["translation_0"]
            )
            current_obj[1]["transcription"] = (
                current_obj[1]["transcription"] + " [turn] " + value["transcription"]
            )
            current_obj[1]["transcription_and_translation"] = (
                current_obj[1]["transcription_and_translation"]
                + value["transcription_and_translation"]
            )

            # append timing information
            # append the begin segment start, based on the current file duration
            current_obj[1][
                "segments_start"
            ] = f"{current_obj[1]['segments_start']} {current_obj[1]['duration']:.2f}"
            current_obj[1][
                "segments_duration"
            ] = f"{current_obj[1]['segments_duration']} {value['duration']:.2f}"
            current_obj[1][
                "segments_channel"
            ] = f"{current_obj[1]['segments_channel']} 0"
            current_obj[1]["duration"] = float(current_obj[1]["duration"]) + float(
                value["duration"]
            )
            current_obj[1]["wav"] = f"{current_obj[1]['wav']}  {value['wav']}"

            # delete the sample from the dataset and key_list
            del dataset[key]
            keys_list.pop(rand_idx)

        # option 3: there's an object and it already overflows the max duration
        else:
            for i in ["transcription", "translation_0"]:
                if len(set(current_obj[1][i].split(" "))) == 2:
                    if list(set(current_obj[1][i].split(" ")))[1] == "[turn]":
                        current_obj[1][i] = ""

            # re-set sample ID
            current_obj[0] = (
                current_obj[0] + f"-{len(current_obj[1]['segments_start'].split())}seg"
            )
            # add the current_obj in the list
            new_dataset[current_obj[0]] = current_obj[1]
            # reset the current_obj
            current_obj = {}

    print(f"printing new JSON object (concatenated) in: {output_file}")
    with open(output_file, mode="w", encoding="utf-8") as json_f:
        json.dump(new_dataset, json_f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
