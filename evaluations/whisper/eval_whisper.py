#!/usr/bin/env/python3
""" Script to use Whisper to Evaluate our data. 
    We evaluate Fisher-CALLHOME datasets and COVOST2
"""

import argparse
import csv
import json
import os
import re
import string
import subprocess

import librosa
import torch
import torchaudio
from speechbrain.utils.bleu import BLEUStats
from speechbrain.utils.metric_stats import ErrorRateStats
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    import xmltodict
    from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
except ImportError:
    err_msg = "The optional dependency sacremoses (or xmltodict) must be installed to run this recipe.\n"
    err_msg += "Install using `pip install sacremoses`.\n"
    raise ImportError(err_msg)


FISHER_DATA_FOLDER = "/folder/to/datasets/fisher_callhome_spanish/data_processed/data"
SAMPLERATE = 16000


# instantiate normalizer and tokenizers
def get_normalizer(lang):
    return MosesPunctNormalizer(lang=lang)


def get_detokenizer(lang):
    return MosesDetokenizer(lang=lang)


def unicode_normalisation(text):
    return str(text)


def strip_accents(text):
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return str(text)


def clean_transcript_translation(text):
    """function to clean output text from Whisper, used in commonvoice and covost"""
    # Unicode Normalization
    words = unicode_normalisation(text)
    # !! Overall cleaning from FISHER-CALLHOME dataset !!
    words = clean_transcript(words)
    # Remove multiple spaces
    words = re.sub(" +", " ", words)
    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip()
    return words


def clean_transcript(transcript: str) -> str:
    """clean the input transcript. Normalize and tokenize"""

    transcript = normalize_punctuation(transcript)
    transcript = clean_transcription(transcript)
    # normalize and tokenizer based on the input language
    normalizer = get_normalizer("en")
    transcript = normalizer.normalize(transcript)
    transcript = remove_punctuation(transcript)

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


def print_bleu_scores(
    file_path, bleu_computer, dataset_name="default", scores_per_conversation=None
):
    """Function to print in a file the new BLEU scores on the aligned file.
    This is done after re-aligning with mwerSegmenter
    scores_per_conversation: needs to be a list, where:
        - key: conversation id
        - text: the BLEU score!

    we also use the bleu_computer and bleu_computer_doc_level to compute new results
    """

    with open(file_path, "w", encoding="utf-8") as w:
        print("=" * 80, file=w)
        print(
            f"\nBLEU score on {dataset_name} - DOCUMENT LEVEL (PER CONVERSATION)",
            file=w,
            end="\n",
        )
        bleu_computer.write_stats(w)
        print("=" * 80, file=w)

        if scores_per_conversation is not None:
            print(f"\nBLEU score per conversation \n", file=w, end="\n")

            [
                print(str(key) + ": " + f"{elem['BLEU']:.3f} ; {elem['EXTRA']}", file=w)
                for key, elem in scores_per_conversation.items()
            ]
            print("=" * 80, file=w)
        print("END", file=w)


def remove_special_tokens(text):
    """remove the special tokens --> turn and xt"""
    words = list(["[turn]", "[xt]"])
    text = re.sub(" +", " ", text)
    text = [word for word in text.split(" ") if word not in words]
    text = " ".join(text)
    text = re.sub(" +", " ", text)
    return text


def extract_reference(gt, language="english"):
    """Function to extract the references from the JSON file and append it in list
    In case there are four references: means it is Fisher-Callhome dev/test sets
    Otherwise, it is standard data with only 1 reference per sample
    """
    lang_dict = {
        "english": "en",
        "french": "fr",
        "german": "de",
        "spanish": "es",
    }

    detokenizer = get_detokenizer(lang=lang_dict[language])

    # check if there's more than one reference, reference 0 is translation_0
    if "translation_1" in gt:
        references = [
            gt["translation_0"],
            gt["translation_1"],
            gt["translation_2"],
            gt["translation_3"],
        ]
    else:
        references = [gt["translation_0"]]

    # detokenize and collect the outputs
    ground_truth = []
    for reference in references:
        # clean the text
        reference = remove_special_tokens(reference)
        detokenized_translation = detokenizer.detokenize(reference.strip().split(" "))
        ground_truth.append([detokenized_translation])
    return ground_truth


import ipdb


def main(args):
    """main function, see parse_arguments"""

    # get CLI input
    output_folder = os.path.join(args.output_folder)
    task = args.task
    source_language = args.source_language

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make output folder if not present
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    dataset_name = args.input_json_file.split("/")[-1].split(".json")[0]
    # file where to put the output BLEU scores after re-aligning
    metrics_file = "bleu" if task == "translate" else "wer"
    metrics_file = os.path.join(
        output_folder, f"{metrics_file}_{dataset_name}_whisper.txt"
    )

    # if decoding happened, omit it
    if os.path.isfile(metrics_file):
        print(f"File {metrics_file}, not decoding with Whisper again!")
        return None

    # reading the ground evaluation data, which follows the convention below:
    print("reading the GT data to evaluate BLEU and WER scores with Whisper")
    with open(f"{args.input_json_file}", "r") as f:
        gt_data_raw = json.load(f)

    model_path = f"openai/{args.model_size}"

    # load model, processor and set decoder IDS
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=source_language, task=task
    )

    # dictionary to store when iterate over ground truth keys
    bleu_per_conversation = {}
    # BLEU computer object
    bleu_computer = BLEUStats(merge_words=False)
    wer_computer = ErrorRateStats()

    for key, values in tqdm(
        gt_data_raw.items(), desc=f"pre-processing [{metrics_file}]"
    ):
        # measure performnace per sample
        bleu_computer_local = BLEUStats(merge_words=False)
        wer_computer_local = ErrorRateStats()

        utt_id = key
        audio_path = values["wav"].replace("{data_root}", FISHER_DATA_FOLDER)

        sig = torch.tensor([])
        # wav field from JSON file might contain several wav files = concatenate them!
        for signal in audio_path.replace("  ", " ").split():
            signal, _ = librosa.load(signal, sr=SAMPLERATE)
            sig = torch.cat([sig, torch.tensor(signal)])

        # get the features with processor
        input_features = processor(
            sig, sampling_rate=SAMPLERATE, return_tensors="pt"
        ).input_features
        input_features = input_features.to(device)

        # generate token ids
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )
        # decode token ids to text
        output = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # post-process output
        clean_output = [clean_transcript_translation(output[0])]

        if task == "translate":
            # gather transcript or translation
            references = extract_reference(values, language="english")

            # append BLEU score per aligned output per sample
            bleu_computer.append(key, clean_output, references)
            # compute BLEU score per conversation to print
            bleu_computer_local.append(key, clean_output, references)
            metadata = bleu_computer_local.summarize()
            metadata["EXTRA"] = f"{references[0][0]} | {clean_output[0]}"
            bleu_per_conversation[key] = metadata
        else:
            # detokenize transcript for the given language, remove special chars
            detokenizer = get_detokenizer(lang=values["source_lang"])
            references = remove_special_tokens(values["transcription"])
            references = detokenizer.detokenize(references.strip().split(" "))
            references = [references.split(" ")]
            clean_output = [clean_output[0].split(" ")]

            # compute WER for this sample, and oif higher than 125%, skip it:
            wer_computer_local.append([key], clean_output, references)
            if wer_computer_local.summarize()["WER"] > 125.0:
                continue

            wer_computer.append([key], clean_output, references)

    print("=" * 80)
    print(f"STEP 4: printing output file in {metrics_file}")
    print("=" * 80)

    if task == "translate":
        # print the bleu scores
        print_bleu_scores(
            metrics_file,
            bleu_computer,
            dataset_name=dataset_name,
            scores_per_conversation=bleu_per_conversation,
        )
    else:
        with open(metrics_file, "w", encoding="utf-8") as w:
            wer_computer.write_stats(w)

    print("Finished printing the new scores after using Whisper!")
    print("=" * 80)
    print("END")
    print("=" * 80)

    return None


def parse_arguments():
    """function to parse input arguments from command-line"""
    parser = argparse.ArgumentParser(
        prog="perform ASR and ST with Whisper",
        description="perform ASR and ST with Whisper",
    )

    parser.add_argument(
        "--input_json_file",
        "-i",
        required=True,
        type=str,
        help="input JSON file with the data to be evaluated",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        required=True,
        type=str,
        help="Output folder where to store all the results and new files",
    )
    parser.add_argument(
        "--model_size",
        required=True,
        type=str,
        default="whisper-tiny",
        choices=[
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large",
        ],
        help="Select the model size of Whisper you want to use (tiny, base or small)",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Select the model size of Whisper you want to use",
    )
    parser.add_argument(
        "--source_language",
        type=str,
        default="english",
        choices=["english", "french", "spanish", "german"],
        help="Select the source language, Important! Whisper: XX -->EN",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
