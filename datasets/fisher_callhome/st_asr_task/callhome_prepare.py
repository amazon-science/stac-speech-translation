"""
Data preparation for CALLHOME-TRANSLATION CORPUS 

- Script for the dual task (ASR + ST)

DATASET: -CALLHOME Spanish Speech               
 - https://catalog.ldc.upenn.edu/LDC96T17
 - https://catalog.ldc.upenn.edu/LDC96S35

Author
-----
* Zuluaga-Gomez, Juan 2023
"""

import json
import logging
import os
import re
import string
import subprocess
from dataclasses import dataclass, field
from typing import List

import torch
import torchaudio
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend
from tqdm import tqdm

try:
    from sacremoses import MosesPunctNormalizer, MosesTokenizer
except ImportError:
    err_msg = (
        "The optional dependency sacremoses must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install sacremoses`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)
check_torchaudio_backend()

es_normalizer = MosesPunctNormalizer(lang="es")
en_normalizer = MosesPunctNormalizer(lang="en")
en_tokenizer = MosesTokenizer(lang="en")

SAMPLE_RATE = 16000


@dataclass
class TDF:
    """
    channel: int
        channel of utterance
    start: int
        start time of utterance
    end: int
        end time of utterance
    transcript: str
        transcript of utteranc
    """

    channel: int
    start: int
    end: int
    transcript: str


@dataclass
class Data:
    """
    each data contains a transcription and a translation for train set
    four translations for dev, dev2, test set
    """

    uid: str = ""
    wav: str = ""
    transcription: str = ""
    duration: float = 0
    translations: List[str] = field(default_factory=lambda: [])


def prepare_only_callhome_spanish(
    data_folder: str, save_folder: str, save_suffix: str = "data", device: str = "cpu"
):
    """
    Prepares the json files for the Mini Callhome-Spanish ST dataset.
    Arguments
    ---------
    data_folder : str
        Path to the folder where the Callhome-Spanish ST dataset is stored.
    save_folder: str:
        Path of train/valid/test specification file will be saved.
    Example
    -------
    >>> data_folder = '/path/to/callhome'
    >>> save_foler = 'data'
    >>> prepare_only_callhome_spanish(data_folder, save_folder)
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # If the dataset doesn't exist yet, terminate the whole program
    # speech_folder = os.path.join(f"{data_folder}/LDC2010S01/data/speech")
    speech_folder = os.path.join(
        f"{data_folder}/LDC96T17/ch_sp/callhome/spanish/speech/"
    )
    transcription_folder = os.path.join(
        f"{data_folder}/LDC96T17/callhome_spanish_trans_970711/transcrp/"
    )

    if check_folders(speech_folder, transcription_folder) is not True:
        logger.error(
            "Speech or transcription directories are missing or not properly organised within the speech data dir"
            "Typical format is LDC96T17/ch_sp/callhome/spanish/speech and LDC96T17/callhome_spanish_trans_970711/transcrp"
        )
        return

    # Subsets of the Callhome dataset
    datasets = ["devtest", "evltest", "train"]

    corpus_path = f"{save_folder}/fisher-callhome-corpus"
    download_translations(path=corpus_path)

    make_data_splits(
        f"{corpus_path}/mapping"
    )  # make splitted data list from mapping files

    for dataset in datasets:
        if not os.path.exists(f"{save_folder}/callhome-{dataset}/wav"):
            os.makedirs(f"{save_folder}/callhome-{dataset}/wav")

        if skip(save_folder, dataset, save_suffix):
            logger.info(
                f"Skipping preparation of {dataset}, completed in previous run."
            )
            continue

        # get file lists
        transcription_files = get_transcription_files_by_dataset(
            dataset, transcription_folder=transcription_folder
        )

        # extract all transcriptions from files
        extracted_transcriptions = {}
        for transcription_file in transcription_files:
            filename = transcription_file.split("/")[-1].split(".")[0]
            extracted_transcriptions[filename] = extract_transcription(
                transcription_file
            )

        # concate short utterance via mapping file
        concated_data = concate_transcriptions_by_mapping_file(
            speech_folder=speech_folder,
            mapping_file_path=f"{corpus_path}/mapping/callhome_{dataset}",
            extracted_transcriptions=extracted_transcriptions,
        )

        # get translation through callhome-corpus
        translation_path = f"{corpus_path}/corpus/ldc/callhome_{dataset}.en"
        translations = get_translations_from_path(translation_path)
        concated_data = insert_translation_into_existing_dataset(
            data=concated_data, translations=translations
        )

        # filter out empty or long transcription/translation
        concated_data = list(
            filter(lambda data: 0 < len(data.transcription) < 400, concated_data)
        )

        concated_data = list(
            filter(
                lambda data: 0 < len(data.translations[0]) < 400,
                concated_data,
            )
        )

        # ignore empty or long utterances
        concated_data = list(filter(lambda data: 0 < data.duration < 30, concated_data))

        # sort by utterance id
        concated_data = sorted(concated_data, key=lambda data: data.uid)

        # store transcription/translation/wav files
        data_dict_asr = {}
        data_dict_st = {}
        for data in tqdm(concated_data, desc=f"pre-processing [{dataset}]"):
            wav_save_path = f"{save_folder}/callhome-{dataset}/wav/{data.uid}.wav"
            # prepare audio files
            wav_information = data.wav.split(" ")

            if not os.path.exists(f"{wav_save_path}"):
                segment_audio(
                    audio_path=wav_information[0],
                    channel=int(wav_information[1]),
                    start=int(wav_information[2]),
                    end=int(wav_information[3]),
                    save_path=wav_save_path,
                    sample_rate=SAMPLE_RATE,
                    device=device,
                )

            # prepare json file for ASR and ST tasks
            data_dict_asr[data.uid + "-asr"] = {
                "wav": f"{wav_save_path}",
                "source_lang": "es",
                "target_lang": "es",
                "duration": data.duration,
                "task": "transcription",
                "transcription": data.transcription,
                "translation_0": data.translations[0],
                "transcription_and_translation": f"{data.transcription}\n{data.translations[0]}",
            }
            data_dict_st[data.uid + "-st"] = {
                "wav": f"{wav_save_path}",
                "source_lang": "es",
                "target_lang": "en",
                "duration": data.duration,
                "task": "translation",
                "transcription": data.transcription,
                "translation_0": data.translations[0],
                "transcription_and_translation": f"{data.transcription}\n{data.translations[0]}",
            }

        # save json ASR and ST tasks
        json_path = f"{save_folder}/callhome-{dataset}/{save_suffix}-asr.json"
        with open(json_path, "w", encoding="utf-8") as data_json:
            json.dump(data_dict_asr, data_json, indent=2, ensure_ascii=False)
        json_path = f"{save_folder}/callhome-{dataset}/{save_suffix}-st.json"
        with open(json_path, "w", encoding="utf-8") as data_json:
            json.dump(data_dict_st, data_json, indent=2, ensure_ascii=False)

        logger.info(f"{json_path} successfully created!")


def skip(save_folder: str, dataset: str, save_suffix: str) -> bool:
    """Detect when callhome data preparation can be skipped"""
    is_skip = True

    if not os.path.isfile(f"{save_folder}/callhome-{dataset}/{save_suffix}-asr.json"):
        is_skip = False

    return is_skip


def check_folders(*folders) -> bool:
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def get_data_list(path: str) -> str:
    with open(path, "r", encoding="utf-8") as data_file:
        return data_file.readlines()


def extract_transcription(transcription_path: str) -> List[TDF]:
    """Extract transcriptions from given file"""
    extracted_transcriptions = []

    with open(transcription_path, "r", encoding="ISO-8859-1") as transcription_file:
        transcriptions = transcription_file.readlines()

        for transcription in transcriptions:
            transcription_fields = transcription.strip().split(" ")

            # remove samples that are not complete:
            if len(transcription_fields) < 4:
                continue

            # channel in position 2
            channel = transcription_fields[2]
            if "A" in channel:
                channel = 0
            elif "B" in channel:
                channel = 1
            else:
                continue

            start = float(transcription_fields[0]) * 100
            end = float(transcription_fields[1]) * 100
            start = int(start)
            end = int(end)

            transcript = " ".join(transcription_fields[3:])

            cleaned_transcript = clean_transcription(transcript)
            extracted_transcriptions.append(
                TDF(
                    channel=channel,
                    start=start,
                    end=end,
                    transcript=cleaned_transcript,
                )
            )

    return extracted_transcriptions


def concate_transcriptions_by_mapping_file(
    speech_folder: str,
    mapping_file_path: str,
    extracted_transcriptions: List[TDF],
) -> List[Data]:
    """return concated transcriptions from the given mapping file"""

    with open(mapping_file_path, "r", encoding="utf-8") as callhome_mapping_file:
        callhome_mapping = callhome_mapping_file.readlines()
        utterances = []

        # files are stored in folders for each subset.
        subset_folder_name = mapping_file_path.split("/")[-1].split("_")[-1]

        for callhome_mapping_line in callhome_mapping:
            callhome_mapping_line = callhome_mapping_line.strip()
            callhome_mapping_line = callhome_mapping_line.split(" ")
            uid = callhome_mapping_line[0]
            need_to_be_concate_lines = callhome_mapping_line[1].split("_")
            need_to_be_concate_lines = list(map(int, need_to_be_concate_lines))

            selected_transcription = extracted_transcriptions[uid]
            # concate multiple transcripts
            if len(need_to_be_concate_lines) > 1:
                # index shift one is because id is count from 1 in file however, list start from 0
                concated_transcripts = selected_transcription[
                    need_to_be_concate_lines[0] - 1 : need_to_be_concate_lines[-1]
                ]
                concated_transcripts = list(
                    map(lambda tdf: tdf.transcript, concated_transcripts)
                )
                concated_transcripts = " ".join(concated_transcripts)

                start = selected_transcription[need_to_be_concate_lines[0] - 1].start
                end = selected_transcription[need_to_be_concate_lines[-1] - 1].end
            else:
                concated_transcripts = selected_transcription[
                    need_to_be_concate_lines[-1] - 1
                ].transcript
                start = selected_transcription[need_to_be_concate_lines[-1] - 1].start
                end = selected_transcription[need_to_be_concate_lines[-1] - 1].end

            # clean up
            concated_transcripts = normalize_punctuation(concated_transcripts)
            concated_transcripts = es_normalizer.normalize(concated_transcripts)

            channel = selected_transcription[need_to_be_concate_lines[0] - 1].channel
            channel_symbol = "B" if channel == 1 else "A"
            uttrance_id = f"{uid}-{channel_symbol}-{start:06d}-{end:06d}"

            # get the folder of subset:
            utterances.append(
                Data(
                    uid=uttrance_id,
                    transcription=concated_transcripts,
                    wav=f"{speech_folder}/{subset_folder_name}/{uid}.wav {channel} {start} {end}",
                    duration=(end - start) / 100,
                )
            )

        return utterances


def segment_audio(
    audio_path: str,
    channel: int,
    start: int,
    end: int,
    save_path: str,
    sample_rate: int = 16000,
    device: str = "cpu",
):
    """segment and resample audio"""

    start = int(start / 100 * 8000)
    end = int(end / 100 * 8000)
    num_frames = end - start

    data, _ = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)

    resampler = Resample(orig_freq=8000, new_freq=sample_rate).to(device=device)

    data = resampler(data)
    data = torch.unsqueeze(data[channel], 0)

    torchaudio.save(save_path, src=data, sample_rate=sample_rate)


def get_transcription_files_by_dataset(
    dataset: str, transcription_folder: str
) -> List[str]:
    """return paths of transcriptions from the given data set and the path of all of transcriptions"""
    train_set = get_data_list(f"splits/{dataset}")
    transcription_train_set = list(
        map(lambda path: path.split(".")[0].strip(), train_set)
    )
    transcription_train_set = list(
        map(lambda path: f"{path}.txt", transcription_train_set)
    )

    transcription_files = get_all_files(
        transcription_folder, match_or=transcription_train_set
    )

    return transcription_files


def get_translations_from_path(translation_path: str) -> List[str]:
    """ "return translations from the given path"""
    extracted_translations = []
    with open(translation_path, "rb") as translations_file:
        original_translations = translations_file.readlines()

        for translation in original_translations:
            translation = translation.replace(b"\r", b"")
            translation = translation.decode("utf-8")

            translation = clean_translation(translation)
            translation = normalize_punctuation(translation)
            translation = en_normalizer.normalize(translation)
            translation = remove_punctuation(translation)
            translation = en_tokenizer.tokenize(translation)

            translation = " ".join(translation)
            extracted_translations.append(translation)

    return extracted_translations


def insert_translation_into_existing_dataset(
    data: List[Data], translations: List[str]
) -> List[Data]:
    """insert corresponding translation to given data"""

    for index in range(len(data)):
        corresponding_translation = translations[index]
        data[index].translations.append(corresponding_translation)

    return data


def download_translations(path: str):
    repo = "https://github.com/joshua-decoder/fisher-callhome-corpus.git"

    if not os.path.isdir(path):
        logger.info(f"Translation file not found. Downloading from {repo}.")
        subprocess.run(["git", "clone", repo])
        subprocess.run(["mv", "fisher-callhome-corpus", f"{path}"])


def make_data_splits(
    mapping_folder: str = "../data/fisher-callhome-corpus/mapping",
):
    """make data split from mapping file"""
    callhome_splits = ["devtest", "evltest", "train"]

    if not os.path.exists("splits/devtest"):
        if not os.path.exists("splits/"):
            os.mkdir("splits")

        for callhome_split in callhome_splits:
            split = set()
            with open(
                f"{mapping_folder}/callhome_{callhome_split}", "r", encoding="utf-8"
            ) as callhome_file, open(
                f"./splits/{callhome_split}", "a+", encoding="utf-8"
            ) as split_file:
                callhome_file_lines = callhome_file.readlines()

                for callhome_file_line in callhome_file_lines:
                    callhome_file_line = callhome_file_line.strip()
                    callhome_file_id = callhome_file_line.split(" ")[0]
                    split.add(callhome_file_id)

                split = sorted(list(split))
                for file_id in split:
                    split_file.write(f"{file_id}\n")


def remove_punctuation(text: str) -> str:
    """remove punctuation from given string"""

    # remove punctuation except apostrophe
    text = text.replace("<space>", "spacemark")
    text = text.replace("'", "apostrophe")

    # based on the definition of [[:punct]]
    punctuation = r"[{}]".format(string.punctuation)

    text = re.sub(punctuation, "", text)
    text = text.replace("spacemark", "<space>")
    text = text.replace("apostrophe", "'")

    # remove consecutive commas and spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"\s+$", "", text)

    return text


def normalize_punctuation(text: str) -> str:
    """remove punctuation from given string"""

    # remove brachets and inside
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^]]+\]", " ", text)

    # normalize punctuation
    text = re.sub(r"_", "", text)
    text = re.sub(r"`", "'", text)  # for En
    text = re.sub(r"´", "'", text)  # for En
    text = re.sub(r"\¨", "'", text)  # I¨m -> I'm etc.

    # remove noisy parts
    text = re.sub(r"noise", "", text)
    text = re.sub(r"laughter", "", text)
    text = re.sub(r"background noise", "", text)
    text = re.sub(r"background speech", "", text)

    # fisher_train
    text = re.sub(r"i\/he", "i", text)
    text = re.sub(r"i\/she", "i", text)
    text = re.sub(r" \/\?", "\\?", text)
    text = re.sub(r" \/ ", " ", text)
    text = re.sub(r"a\/c", "", text)
    text = re.sub(r"stay\/", "stay", text)
    text = re.sub(r"boys\/", "boys", text)
    text = re.sub(r"right\/", "right", text)
    text = re.sub(r"follow\/", "follow", text)
    text = re.sub(r"Jose\/Josefina", "Jose", text)
    text = re.sub(r"welfare\/foreign", "welfare", text)
    text = re.sub(r"\<foreign lang=\"English\"", "", text)
    text = re.sub(r"\/foreign/", "", text)
    text = re.sub(r"\<plural\>", "", text)
    text = re.sub(r"\<barely makes any sense\>", "", text)
    text = re.sub(r"\<kind of a weird phrase\>", "", text)
    text = re.sub(r"\<last word does not fit there\>", "", text)
    text = re.sub(r"\<players with the meaning of singers\>", "", text)
    text = re.sub(r"\<this phrase barely made any sense whatsoever\>", "", text)
    text = re.sub(
        r"\<colorcito does not exist as a word so I have no ideea what he means about that\>",
        "",
        text,
    )
    text = re.sub(r"\<foreign", "", text)
    text = re.sub(r"foreign\>", "", text)

    # fisher_dev
    text = re.sub(r"her\/his", "her", text)
    text = re.sub(r"o\/", "o", text)
    text = re.sub(r"co\/", "co", text)
    text = re.sub(r"L \/ ", "", text)
    text = re.sub(r"\<\?\?\?\>", "", text)
    text = re.sub(r"\<from Texas\>", "", text)
    text = re.sub(r"\<weird phrase\>", "", text)
    text = re.sub(r"\<weird phrase\>", "", text)
    text = re.sub(r"\<this makes no sense\>", "", text)
    text = re.sub(r"Salvador\>", "Salvador", text)

    # fisher_dev 2
    text = re.sub(r"A\/C", "", text)
    text = re.sub(r"She\/he", "She", text)
    text = re.sub(r"you\/he", "you", text)
    text = re.sub(r"you\/she", "you", text)
    text = re.sub(r"Um\/", "Um", text)
    text = re.sub(r"name\/", "name", text)
    text = re.sub(r"American\/", "American", text)
    text = re.sub(r"\<\?\>", "", text)
    text = re.sub(r"\<metaphoric meaning\>", "", text)
    text = re.sub(r"\<missing text \? \>", "", text)
    text = re.sub(
        r"\<broken phrase but I tried to guess what would it mean if it was complete\>",
        "",
        text,
    )

    # fisher_test
    text = re.sub(r"she\/he", "she", text)
    text = re.sub(r"her\/him", "her", text)
    text = re.sub(r"is\/", "is", text)
    text = re.sub(r"and\/or", "and", text)
    text = re.sub(r"Then\/Well", "Then", text)
    text = re.sub(r"fine\/well", "fine", text)
    text = re.sub(r"Likewise\/Equally", "Likewise", text)
    text = re.sub(r"boyfriend\/girlfriend", "boyfriend", text)
    text = re.sub(r"living room \/ dining room", "living room", text)
    text = re.sub(r"\<very bad phrase\>", "", text)
    text = re.sub(r"\<poorly written phrase\>", "", text)
    text = re.sub(r"\<this phrase barely even made sense\>", "", text)
    text = re.sub(
        r"\<very poorly written phrase but I think this is what was supposed to mean\>",
        "",
        text,
    )
    text = re.sub(r"what\)\)", "what", text)

    # remove noisy punctuation
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\<", " ", text)
    text = re.sub(r"\>", " ", text)
    text = re.sub(r"\[", " ", text)
    text = re.sub(r"\]", " ", text)
    text = re.sub(r"\{", " ", text)
    text = re.sub(r"\}", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\;", " ", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"=", " ", text)
    text = re.sub(r"\·", " ", text)

    text = re.sub(r"^\.\s*$", "", text)  # only period sentence
    text = re.sub(r"^\?\s*$", "", text)  # only question mark sentence
    text = re.sub(r"\s+", " ", text)  # remove consecutive whitespaces

    # remove the first and last whitespaces
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"\s+$", "", text)

    text = text.lstrip()

    return text


def clean_transcription(transcription: str) -> str:
    """
    original: https://github.com/jamfly/AlloST/blob/main/egs/fisher_callhome_spanish/st1/local/fsp_data_prep.sh

    clean a given transcription and return a cleaned transcription
    """

    transcription = transcription.replace("</", "lendarrow")
    transcription = transcription.replace("<", "larrow")
    transcription = transcription.replace(">", "rarrow")
    # add the tags of callhome
    transcription = transcription.replace("[[", "larrow")
    transcription = transcription.replace("[", "larrow")
    transcription = transcription.replace("{", "larrow")
    transcription = transcription.replace("]]", "rarrow")
    transcription = transcription.replace("]", "rarrow")
    transcription = transcription.replace("}", "rarrow")

    punctuation = r"[{}]".format(string.punctuation)
    transcription = re.sub(punctuation, "", transcription)

    transcription = transcription.replace("larrow", "<")
    transcription = transcription.replace("rarrow", ">")
    transcription = transcription.replace("lendarrow", "</")

    transcription = transcription.replace("Á", "á")
    transcription = transcription.replace("Í", "í")
    transcription = transcription.replace("Ó", "ó")
    transcription = transcription.replace("Ú", "ú")
    transcription = transcription.replace("¨", "")
    transcription = transcription.replace("·", "")
    transcription = transcription.replace("´", "")
    transcription = transcription.replace("¿", "")
    transcription = transcription.replace("¡", "")
    transcription = transcription.replace("N", "n")

    transcription = transcription.lower()

    transcription = remove_labels(transcription)

    return transcription


def clean_translation(translation: str) -> str:
    """clean a given translation and returne a cleaned translation"""
    translation = translation.strip()
    translation = translation.lower()

    translation = translation.replace("¿", "")
    translation = translation.replace("¡", "")

    return translation


def remove_labels(transcription: str):
    """remove label such as <laugh> from transcript"""

    # remove everything between <>
    transcription = re.sub(r"\<[^<>]*\>", "", transcription)

    transcription = re.sub(r"<\s*[/]*\s*\s*for[ei][ei]g[nh]\s*\w*>", "", transcription)
    transcriptions = re.findall(r"<lname>\([^<]*\)<\/lname>", transcription)

    if len(transcriptions) > 0:
        transcription = transcriptions[0]

    transcription = re.sub(r"<lname[\/]*>", "", transcription)
    transcription = re.sub(r"<laugh>", "", transcription)
    transcription = re.sub(r"<\/laugh>", "", transcription)
    transcription = re.sub(r"<\s*cough[\/]*>", "[noise]", transcription)
    transcription = re.sub(r"<sneeze[\/]*>", "[noise]", transcription)
    transcription = re.sub(r"<breath[\/]*>", "[noise]", transcription)
    transcription = re.sub(r"<lipsmack[\/]*>", "[noise]", transcription)
    transcription = re.sub(r"<background>", "", transcription)
    transcription = re.sub(r"<\/background>", "", transcription)
    transcription = re.sub(r"<[/]?background[/]?>", "[noise]", transcription)
    transcription = re.sub(r"<laugh>", "", transcription)
    transcription = re.sub(r"<\/laugh>", "", transcription)
    transcription = re.sub(r"<[/]?laugh[/]?>", "[laughter]", transcription)
    transcription = re.sub(r"<foreign langenglishhip hop", "", transcription)
    transcription = re.sub(r"<foreign langenglishonline", "", transcription)
    transcription = re.sub(r"<foreign langenglish", "", transcription)
    transcription = re.sub(r"</foreign", "", transcription)
    transcription = re.sub(r"<[/]?foreing\s*\w*>", "", transcription)
    transcription = re.sub(r"</b", "", transcription)
    transcription = re.sub(r"<foreign langengullís>", "", transcription)
    transcription = re.sub(r"foreign>", "", transcription)
    transcription = re.sub(r">", "", transcription)

    is_match = re.search(r"\(\)", transcription)

    if is_match is not True:
        transcription = re.sub(r"\[noise\]", "", transcription)
        transcription = re.sub(r"\[laughter\]", "", transcription)
        transcription = re.sub(r"^\s\s*|\s\s*$", "", transcription)
        transcription = re.sub(r"^\s\s*", " ", transcription)

    return transcription


if __name__ == "__main__":
    data_folder = "/mnt/md0/user_jamfly/CORPUS"
    save_folder = "data"
    device = "cuda:0"

    prepare_only_callhome_spanish(data_folder, save_folder, device=device)
