"""
Data preparation for MSLT dataset.

Author
------
 * Juan Zuluaga-Gomez,
"""
import argparse
import csv
import glob
import json
import logging
import os
import re
import string
import sys
import unicodedata

import torchaudio
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend

try:
    from sacremoses import MosesPunctNormalizer, MosesTokenizer
except ImportError:
    err_msg = (
        "The optional dependency sacremoses must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install sacremoses`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# instantiate normalizer and tokenizers
_NORMALIZER = {
    "en": MosesPunctNormalizer(lang="en"),
    "de": MosesPunctNormalizer(lang="de"),
    "fr": MosesPunctNormalizer(lang="fr"),
    "zh": MosesPunctNormalizer(lang="zh"),
    "ja": MosesPunctNormalizer(lang="ja"),
}
_TOKENIZERS = {
    "en": MosesTokenizer(lang="en"),
    "de": MosesTokenizer(lang="de"),
    "fr": MosesTokenizer(lang="fr"),
    "zh": MosesTokenizer(lang="zh"),
    "ja": MosesTokenizer(lang="ja"),
}
VERSION_LANGUAGES = {
    "1": ["en", "de", "fr"],
    "1_1": ["en", "ja", "zh"],
}
VERSION_SUFFIX = {
    "1_dev": "20160616",
    "1_test": "20160516",
    "1_1": "20170914",
}

# target sample rate!
SAMPLE_RATE = 16000


def prepare_mslt_dataset(
    version,
    data_folder,
    save_folder,
    accented_letters=True,
    duration_threshold=30,
    skip_prep=False,
):
    """
    Prepares the JSON files for the MSLT dataset.
    Download: https://github.com/MicrosoftTranslator/MSLT-Corpus 
    Arguments
    ---------
    version : str
        It should be either "1" or "1_1". This will change the languages to prepare!
    data_folder : str
        Path to the folder where the original dataset is stored.
        This path should include the direct to Data: /downloads/v1/MSLT_Corpus/Data/
    save_folder : str
        The directory where to store the JSON files.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold: float, optional
        Defines the max audio duration we want to use
    skip_prep: bool
        If True, skip data preparation.
    Example
    -------
    >>> from recipes.mslt.mslt_prepare import prepare_mslt_dataset
    >>> version = '1'
    >>> data_folder = '/datasets/mslt/v1/'
    >>> save_folder = 'exp/mslt_v1/'
    >>> accented_letters = True
    >>> duration_threshold = 45
    >>> prepare_mslt_dataset( \
                 version, \
                 data_folder, \
                 save_folder, \
                 accented_letters, \
                 duration_threshold, \
                 )
    """

    if skip_prep:
        return
    if version != "1" and version != "1_1":
        print(f"Please, pass a version: either 1 or 1_1, you passed {version}")
        return

    # set save_folder path
    save_folder = os.path.join(save_folder, f"v_{version}")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get the files to create vased on the defined version
    save_json_files = {}
    for subset in ["Dev", "Test"]:
        for source_lang in VERSION_LANGUAGES[version]:
            for target_lang in VERSION_LANGUAGES[version]:
                # prepare file id based on the version and source--> target lang
                # if source_lang == target_lang it means ASR
                file_id = (
                    f"mslt_{version}v__{subset.lower()}_{source_lang}_{target_lang}"
                )
                save_json_files[file_id] = os.path.join(save_folder, file_id)

    # If JSON files already exists, we skip the data preparation
    if skip(save_json_files):
        msg = f"All the files for MSLT version: {version} already exists, skipping data preparation!"
        logger.info(msg)
        return

    for json_id in save_json_files:
        json_path_file = save_json_files[json_id]

        # check if json file is already present and skip it
        if not os.path.isfile(json_path_file + ".json"):
            create_json(
                data_folder,
                json_id,
                json_path_file,
                version,
                accented_letters,
                duration_threshold,
            )
        else:
            logger.info(f"{json_path_file}.json already present, skipping it...")


def skip(save_json_files):
    """
    Detects if the MSLT data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking all folders
    skip = False
    if all([os.path.isfile(j + ".json") for i, j in save_json_files.items()]):
        skip = True
    return skip


def create_json(
    data_folder,
    json_file,
    json_path_file,
    version,
    accented_letters=False,
    duration_threshold=30,
):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    json_file : str
        ID for the json file, here we extract the source and target lang
    json_path_file : str
        path where to sttore the json file
    version : str
        Version of the dataset to use.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    duration_threshold: float, optional
        Defines the maximum duration threshold for appending samples
    Returns
    -------
    None
    """

    # get the information about the file about to process!
    target_lang = json_file.split("_")[-1]
    source_lang = json_file.split("_")[-2]
    subset = json_file.split("_")[-3]
    subset = subset[0].upper() + subset[1:]

    if version == "1":
        version_folder = (
            VERSION_SUFFIX["1_dev"] if subset == "Dev" else VERSION_SUFFIX["1_test"]
        )
    elif version == "1_1":
        version_folder = VERSION_SUFFIX[version]
    folder_id = f"MSLT_{subset}_{source_lang.upper()}_{version_folder}"

    # select the target file suffix depending on target_lang
    # in MSLT, T1 and T2 are transcripts and T3.xx is the translation
    suffix = "T2" if source_lang == target_lang else "T3"

    data_folder = os.path.join(data_folder, "MSLT_Corpus", "Data", folder_id)
    # Check if the given files exists
    if not os.path.isdir(data_folder):
        msg = "\t%s doesn't exist, verify your dataset!" % (data_folder)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # Setting torchaudio backend to sox-io (needed to read flac or mp3 files)
    if torchaudio.get_audio_backend() != "sox_io":
        logger.warning("This recipe needs the sox-io backend of torchaudio")
        logger.warning("The torchaudio backend is changed to sox_io")
        torchaudio.set_audio_backend("sox_io")

    logger.info(f"Preparing JSON files for {data_folder}")

    data_dict = {}
    total_duration = 0
    for wavfile in glob.glob(os.path.join(data_folder, "*.wav")):
        # now confirm that this sample has all the annotations
        filename = wavfile.split(".T0.")[0]

        # peak at the wav file to retrieve data
        info = torchaudio.info(wavfile)
        # get duration
        duration = info.num_frames / info.sample_rate

        if info.num_channels > 1:
            logger.info(f"File: {wavfile} has more than one channel...")
            continue
        if info.sample_rate != 16000:
            logger.info(f"File: {wavfile} is not sampled at 16 kHz...")
            continue
        # check that the recording is no longer than threshold
        if duration > duration_threshold:
            logger.info(f"File: {wavfile} exceed threshold ({duration_threshold})")
            continue

        # retrieve the transcript or translation, and join it
        try:
            # Weird bug in the folder organization
            # When the source lang is English, the locales should not be changed
            # when the source_lang is ZH or JP, we need to change the locales
            if target_lang == "zh" and source_lang != "en":
                target_lang_fixed = source_lang_fixed = "ch"
            elif target_lang == "ja" and source_lang != "en":
                target_lang_fixed = source_lang_fixed = "jp"
            else:
                source_lang_fixed, target_lang_fixed = source_lang, target_lang
            lines = open(
                filename + f".{suffix}.{target_lang_fixed}.snt", "r", encoding="utf-16"
            ).readlines()
        except:
            logger.info(f"Problem reading: {filename}.{suffix}.{target_lang_fixed}.snt")
            continue

        words = clean_all_transcript(lines, target_lang, accented_letters)
        if words == False:
            continue

        # in case we have a translation direction, retrieve also the transcripts
        # to append to sample!
        if source_lang != target_lang:
            if source_lang_fixed == "ja":
                source_lang_fixed = "jp"
            elif source_lang_fixed == "zh":
                source_lang_fixed = "ch"

            lines_transcript = open(
                filename + f".T2.{source_lang_fixed}.snt", "r", encoding="utf-16"
            ).readlines()
            words_transcript = clean_all_transcript(
                lines_transcript, source_lang, accented_letters
            )
            if words_transcript == False:
                continue

        # compute total duration
        total_duration += duration
        # define the task and set transcription and translation
        if source_lang == target_lang:
            transcription, translation = words, ""
            task = "transcription"
        else:
            transcription, translation = words_transcript, words
            task = "translation"

        # prepare json file
        snt_id = f"{json_file}_{filename.split('/')[-1]}"
        data_dict[snt_id] = {
            "wav": wavfile,
            "duration": duration,
            "task": task,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "transcription": transcription,
            "translation_0": translation,
            "transcription_and_translation": f"{transcription} \n {translation}",
        }

    # in case the dataset is empty, do not create a file
    if len(data_dict) < 1:
        return
    # Writing the json file
    with open(json_path_file + ".json", mode="w", encoding="utf-8") as json_f:
        json.dump(data_dict, json_f, indent=2, ensure_ascii=False)

    # Final prints
    msg = "%s successfully created!" % (json_path_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(data_dict)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def clean_all_transcript(text, target_lang, accented_letters):
    """this function calls the other functions to clean the input
    transcript or translation
    """
    # join in case there are several lines
    words = " ".join([i.strip() for i in text])

    # Unicode Normalization
    words = unicode_normalisation(words)

    # we omit this pre-process for now
    # !! Language specific cleaning !!
    # words = language_specific_preprocess(target_lang, words)

    # !! Overall cleaning from FISHER-CALLHOME dataset !!
    words = clean_transcript(words, target_lang)

    # Remove accents if specified
    if not accented_letters:
        words = strip_accents(words)
        words = words.replace("'", " ")
        words = words.replace("’", " ")

    # Remove multiple spaces
    words = re.sub(" +", " ", words)

    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip()

    # Getting chars
    chars = words.replace(" ", "_")
    chars = " ".join([char for char in chars][:])

    # Remove too short sentences (or empty):
    if target_lang in ["ja", "ch"]:
        if len(chars) < 3:
            return False
    else:
        if len(words.split(" ")) < 2:
            return False
    return words


def language_specific_preprocess(language, words):
    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.

    if language in ["en", "fr", "it", "rw"]:
        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()

    if language == "de":
        # this replacement helps preserve the case of ß
        # (and helps retain solitary occurrences of SS)
        # since python's upper() converts ß to SS.
        words = words.replace("ß", "0000ß0000")
        words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words).upper()
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        words = words.replace(
            "0000SS0000", "ß"
        )  # replace 0000SS0000 back to ß as its initial presence in the corpus

    if language == "fr":
        # Replace J'y D'hui etc by J_ D_hui
        words = words.replace("'", " ")
        words = words.replace("’", " ")

    elif language == "es":
        # Fix the following error in dataset large:
        # KeyError: 'The item En noviembre lanzaron Queen Elizabeth , coproducida por Foreign Noi$e . requires replacements which were not supplied.'
        words = words.replace("$", "s")
    return words


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.
    If not, raises an error.
    Returns
    -------
    None
    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """
    files_str = "/clips"
    # Checking clips
    if not os.path.exists(data_folder + files_str):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):
    return str(text)


def strip_accents(text):
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return str(text)


def clean_transcript(transcript: str, language: str) -> str:
    """clean the input transcript. Normalize and tokenize"""
    transcript = clean_transcription(transcript)
    transcript = normalize_punctuation(transcript)

    # normalize and tokenizer based on the input language
    normalizer = _NORMALIZER[language]
    tokenizer = _TOKENIZERS[language]
    transcript = normalizer.normalize(transcript)
    transcript = remove_punctuation(transcript)
    transcript = tokenizer.tokenize(transcript)
    transcript = " ".join(transcript)

    return transcript


def remove_punctuation(text: str) -> str:
    """remove punctuation from given string"""

    # remove punctuation except apostrophe
    text = text.replace("<space>", "spacemark")
    text = text.replace("'", "apostrophe")

    # based on the definition of [[:punct]]
    punctuation = r"[{}]".format(string.punctuation).replace("'", "")

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

    punctuation = r"[{}]".format(string.punctuation).replace("'", "")
    transcription = re.sub(punctuation, "", transcription)

    transcription = transcription.replace("larrow", "<")
    transcription = transcription.replace("rarrow", ">")
    transcription = transcription.replace("lendarrow", "</")

    transcription = transcription.replace("Á", "á")
    transcription = transcription.replace("Í", "í")
    transcription = transcription.replace("Ó", "ó")
    transcription = transcription.replace("Ú", "ú")
    transcription = transcription.replace("¨", " ")
    transcription = transcription.replace("·", " ")
    transcription = transcription.replace("´", " ")
    transcription = transcription.replace("¿", " ")
    transcription = transcription.replace("¡", " ")
    transcription = transcription.replace("N", "n")
    transcription = transcription.replace("N", "n")

    transcription = transcription.lower()

    transcription = remove_labels(transcription)
    transcription = transcription.replace("¿", " ")
    transcription = transcription.replace("¡", " ")

    return transcription


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder", "-i", type=str, required=True, help="input data folder"
    )
    parser.add_argument(
        "--save-folder",
        "-o",
        type=str,
        required=True,
        help="where to store the JSON files",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        required=True,
        choices=("1", "1_1"),
        help="The version of the MSLT dataset to be used.",
    )
    parser.add_argument(
        "--accented-letters",
        "-a",
        type=bool,
        default=True,
        help="This flag allows to keep or remove accents",
    )
    parser.add_argument(
        "--duration-threshold",
        "-t",
        type=float,
        default=30.0,
        help="Max audio duration allowed",
    )
    return parser.parse_args()


def main():
    """main function, you can also import directly"""
    args = get_args()

    prepare_mslt_dataset(
        version=args.version,
        data_folder=args.data_folder,
        save_folder=args.save_folder,
        accented_letters=args.accented_letters,
        duration_threshold=args.duration_threshold,
    )
    msg = f"finished prepareding dataset in: {args.save_folder}"
    logger.info(msg)


if __name__ == "__main__":
    main()
