"""
Data preparation for EN and ES datasets from CommonVoice.

Author
------
 * Juan Zuluaga-Gomez, 2023
"""
import argparse
import os

from common_voice_prepare import prepare_common_voice


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", "-l", type=str, required=True, help="Language to prepare"
    )
    parser.add_argument(
        "--data-folder",
        "-d",
        type=str,
        required=True,
        help="Folder with the CSV and clips",
    )
    parser.add_argument(
        "--save-folder", "-o", type=str, required=True, help="output folder"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=45,
        required=False,
        help="Duration threshold in seconds",
    )
    return parser.parse_args()


def main():
    args = get_args()
    locale = args.language
    data_folder = args.data_folder
    save_folder = args.save_folder

    # get the train/dev/test sets
    train_tsv_file = os.path.join(data_folder, locale, "train.tsv")
    dev_tsv_file = os.path.join(data_folder, locale, "dev.tsv")
    test_tsv_file = os.path.join(data_folder, locale, "test.tsv")
    train_validated_tsv_file = os.path.join(data_folder, locale, "train_validated.tsv")

    # run the preparation script
    prepare_common_voice(
        os.path.join(data_folder, locale),
        os.path.join(save_folder, locale),
        train_tsv_file,
        dev_tsv_file,
        test_tsv_file,
        train_validated_tsv_file,
        accented_letters=True,
        duration_threshold=args.threshold,
        language=locale,
    )

    print(
        f"finished preparing the CommonVoice dataset for {args.language} in: {data_folder}"
    )


if __name__ == "__main__":
    main()
