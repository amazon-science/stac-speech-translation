"""
Script to insert translations from COVOST2 into JSON files for CommonVoice!

Author
------
 * Juan Zuluaga-Gomez, 2023

"""

import argparse
import csv
import json
import re
import string
import unicodedata

try:
    from sacremoses import MosesPunctNormalizer, MosesTokenizer
except ImportError:
    err_msg = (
        "The optional dependency sacremoses must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install sacremoses`.\n"
    raise ImportError(err_msg)


# instantiate normalizer and tokenizers
def get_normalizer(lang):
    return MosesPunctNormalizer(lang=lang)


def get_tokenizer(lang):
    return MosesTokenizer(lang=lang)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json", "-j", type=str, required=True, help="JSON file of the given set"
    )
    parser.add_argument(
        "--tsv-file",
        "-t",
        type=str,
        required=True,
        help="TSV separated file with translations",
    )
    parser.add_argument(
        "--target-lang",
        "-l",
        type=str,
        required=True,
        help="the locale of the target language, so we can do proper filtering",
    )
    parser.add_argument(
        "--keep-accents",
        "-a",
        type=bool,
        default=True,
        help="This flag allows to keep or remove accents",
    )
    return parser.parse_args()


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


def main():
    """main function"""
    args = get_args()

    locale = args.target_lang
    accented_letters = args.keep_accents

    print(f"Converting to JSON: {args.tsv_file}")
    # load the JSON data from sample_wav
    with open(args.input_json, "r") as f:
        json_dataset = json.load(f)

    with open(args.tsv_file, "r") as f:
        tsv_dataset = {}
        for line in f:
            line = line.strip().split("\t")
            if line[0] == "path":
                continue
            utt_id = line[0].split(".mp3")[0]
            translation = line[2]
            tsv_dataset[utt_id] = translation

    new_dataset = {}
    cnt_error = 0
    # iterate over the TSV dataset! Which is the one that we care about
    for key in tsv_dataset:
        try:
            target_json = json_dataset[key]
            translation = tsv_dataset[key]
        except:
            print(f"Key: {key} not present in CommonVoice dataset")
            cnt_error += 1
        # Unicode Normalization
        words = unicode_normalisation(translation)

        # we omit this pre-process for now
        # !! Language specific cleaning !!
        # words = language_specific_preprocess(locale, words)

        # !! Overall cleaning from FISHER-CALLHOME dataset !!
        words = clean_transcript(words, locale)

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
        if locale in ["ja", "ch"]:
            if len(chars) < 3:
                continue
        else:
            if len(words.split(" ")) < 3:
                continue

        # update the values in the json object
        target_json["task"] = "translation"
        target_json["target_lang"] = locale
        target_json["translation_0"] = words

        # append object to new dataset!
        new_dataset[key] = target_json

    print(f"{cnt_error} samples (out of {len(tsv_dataset)}) that could not be prepared")

    # Writing the json file
    json_file = args.tsv_file.replace(".tsv", ".json")
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(new_dataset, json_f, indent=2, ensure_ascii=False)

    # Final prints
    print(f"successfully created {json_file}!")


if __name__ == "__main__":
    main()
