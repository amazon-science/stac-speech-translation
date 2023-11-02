"""
Join 2 or more JSON files!.

Author
------
 * Juan Zuluaga-Gomez, 2023
"""
import json
import sys


def main():
    args = sys.argv
    files_to_join = args[1:-1]
    output_file = args[-1]

    output_json_file = {}
    for json_file in files_to_join:
        # load JSON file
        with open(json_file, "r") as f:
            dataset = json.load(f)

        # insert a new field to train the Tokenizer
        for key in dataset.keys():
            if "transcription_and_translation" not in dataset[key]:
                dataset[key][
                    "transcription_and_translation"
                ] = f"{dataset[key]['transcription']} \n {dataset[key]['translation_0']}"

            if len(dataset[key]["transcription_and_translation"]) <= 1:
                dataset[key][
                    "transcription_and_translation"
                ] = f"{dataset[key]['transcription']} \n {dataset[key]['translation_0']}"

        # append the new JSON output
        output_json_file.update(dataset)

    print(f"printing new JSON object (concatenated) in: {output_file}")
    with open(output_file, mode="w", encoding="utf-8") as json_f:
        json.dump(output_json_file, json_f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
