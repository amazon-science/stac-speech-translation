"""
Data preparation.
Download: https://voice.mozilla.org/en/datasets
Author
------
Titouan Parcollet
Luca Della Libera 2022
Pooneh Mousavi 2022
Juan Zuluaga Gomez 2023
"""
import csv
import json
import logging
import os
import re
import string
import unicodedata

import soundfile as sf
import torch
import torchaudio
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend
from tqdm.contrib import tzip

try:
    from sacremoses import MosesPunctNormalizer, MosesTokenizer
except ImportError:
    err_msg = (
        "The optional dependency sacremoses must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install sacremoses`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


# instantiate normalizer and tokenizers
def get_normalizer(lang):
    return MosesPunctNormalizer(lang=lang)


def get_tokenizer(lang):
    return MosesTokenizer(lang=lang)


# target sample rate!
SAMPLE_RATE = 16000


def prepare_common_voice(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    train_validated_tsv_file=None,
    accented_letters=False,
    language="en",
    duration_threshold=45,
    skip_prep=False,
):
    """
    Prepares the JSON files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/<language>/
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    train_validated_tsv_file : str, optional
        Path to the Validated Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.
    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> train_validated_tsv_file = '/datasets/CommonVoice/en/train_validated.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 train_validated_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    if train_validated_tsv_file is None:
        train_validated_tsv_file = data_folder + "/train_validated.tsv"
    else:
        train_validated_tsv_file = train_validated_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_json_train = save_folder + "/train.json"
    save_json_dev = save_folder + "/dev.json"
    save_json_test = save_folder + "/test.json"
    save_json_train_validated = save_folder + "/train_validated.json"

    # If JSONs already exists, we skip the data preparation
    if skip(save_json_train, save_json_dev, save_json_test, save_json_train_validated):
        msg = "%s already exists, skipping data preparation!" % (save_json_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_json_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_json_test)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (
            save_json_train_validated
        )
        logger.info(msg)
        return

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)
    # Creating JSON files for {train_validated, train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file, train_validated_tsv_file],
        [save_json_train, save_json_dev, save_json_test, save_json_train_validated],
    )

    for tsv_file, save_json in file_pairs:
        create_json(
            tsv_file,
            save_json,
            data_folder,
            accented_letters,
            language,
            duration_threshold,
        )


def skip(save_json_train, save_json_dev, save_json_test, save_json_train_validated):
    """
    Detects if the Common Voice data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_json_train)
        and os.path.isfile(save_json_dev)
        and os.path.isfile(save_json_test)
        and os.path.isfile(save_json_train_validated)
    ):
        skip = True

    return skip


def create_json(
    orig_tsv_file,
    json_file,
    data_folder,
    accented_letters=False,
    language="en",
    duration_threshold=30,
):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing JSON files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating JSON lists in %s ..." % (json_file)
    logger.info(msg)

    # folder where to store the new wav files:
    wav_folder_path = os.path.join("/".join(json_file.split("/")[:-1]), "clips_wav")
    # Setting the save folder
    if not os.path.exists(wav_folder_path):
        os.makedirs(wav_folder_path)

    # store transcription/wav files
    data_dict = {}

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):
        line = line[0]

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/clips/" + line.split("\t")[1]
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(mp3_path):
            info = sf.info(mp3_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.duration
        # check that the recording is no longer than threshold
        if duration > duration_threshold:
            continue
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Overall cleaning from FISHER-CALLHOME dataset !!
        words = clean_transcript(words, language)

        # we omit this pre-process for now
        # !! Language specific cleaning !!
        # words = language_specific_preprocess(language, words)

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
        if language in ["ja", "ch"]:
            if len(chars) < 3:
                continue
        else:
            if len(words.split(" ")) < 3:
                continue

        # convert the file into wav file, so torchaudio do not have issues!
        wav_save_path = mp3_path.split("/")[-1].replace(".mp3", ".wav")
        wav_save_path = os.path.join(wav_folder_path, wav_save_path.split("/")[-1])

        if not os.path.exists(f"{wav_save_path}"):
            segment_audio(
                audio_path=mp3_path,
                channel=0,
                save_path=wav_save_path,
                sample_rate=SAMPLE_RATE,
                device="cpu",
            )

        # prepare json file
        data_dict[snt_id] = {
            "wav": wav_save_path,
            "duration": duration,
            "task": "transcription",
            "source_lang": language,
            "target_lang": language,
            "transcription": words,
            "translation_0": "",
        }

    # Writing the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(data_dict, json_f, indent=2, ensure_ascii=False)

    # Final prints
    msg = "%s successfully created!" % (json_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def segment_audio(
    audio_path: str,
    channel: int,
    save_path: str,
    sample_rate: int = 16000,
    device: str = "cpu",
):
    """segment and resample audio"""
    data, file_sr = sf.read(audio_path)
    data = torch.tensor(data).float()
    # we need to merge both channels, in case there are two
    if len(data.shape) > 1:
        data = torch.mean(data, dim=0).view(1, data.shape[1])
        print(f"two channels in {audio_path}")

    resampler = Resample(orig_freq=file_sr, new_freq=sample_rate).to(device=device)
    data = resampler(data.view(1, data.shape[0]))
    data = torch.unsqueeze(data[channel], 0)
    # save the file in the given save_path
    torchaudio.save(save_path, src=data, sample_rate=sample_rate)


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

    transcript = normalize_punctuation(transcript)
    transcript = clean_transcription(transcript)

    # normalize and tokenizer based on the input language
    normalizer = get_normalizer(language)
    tokenizer = get_tokenizer(language)
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


# _NORMALIZER = {
#     "en": MosesPunctNormalizer(lang="en"),
#     "de": MosesPunctNormalizer(lang="de"),
#     "fr": MosesPunctNormalizer(lang="fr"),
#     "es": MosesPunctNormalizer(lang="es"),
#     "sv-SE": MosesPunctNormalizer(lang="sv"),
#     "sl": MosesPunctNormalizer(lang="sl"),
#     "et": MosesPunctNormalizer(lang="et"),
#     "it": MosesPunctNormalizer(lang="it"),
#     "pt": MosesPunctNormalizer(lang="pt"),
#     "nl": MosesPunctNormalizer(lang="nl"),
#     "lv": MosesPunctNormalizer(lang="lv"),
#     "cy": MosesPunctNormalizer(lang="cy"),
# }

# _TOKENIZERS = {
#     "en": MosesTokenizer(lang="en"),
#     "de": MosesTokenizer(lang="de"),
#     "fr": MosesTokenizer(lang="fr"),
#     "es": MosesTokenizer(lang="es"),
#     "sv-SE": MosesTokenizer(lang="sv"),
#     "sl": MosesTokenizer(lang="sl"),
#     "et": MosesTokenizer(lang="et"),
#     "it": MosesTokenizer(lang="it"),
#     "pt": MosesTokenizer(lang="pt"),
#     "nl": MosesTokenizer(lang="nl"),
#     "lv": MosesTokenizer(lang="lv"),
#     "cy": MosesTokenizer(lang="cy"),
# }
