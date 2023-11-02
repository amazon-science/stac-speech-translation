#!/usr/bin/env/python3
""" 
Dataio functions read by the main train_multitask.py file

Author
------
 * Juan Zuluaga-Gomez, 2023
"""
import csv
import json
import logging
import os
import sys

import ipdb
import speechbrain as sb
import torch
from sacremoses import MosesDetokenizer

FISHER_DATA_FOLDER = (
    "/folder/to/datasets/fisher_callhome_spanish/data_processed/data"
)


def get_detokenizer(language):
    """function to get a MosesDetokenizer based on the input language locale"""
    # detokenizer = MosesDetokenizer(lang=language)
    return MosesDetokenizer(lang=language)


# Potentially, it kind be expanded to more languages
_DETOKENIZERS = {
    "en": get_detokenizer(language="en"),
    "es": get_detokenizer(language="es"),
    "de": get_detokenizer(language="de"),
    "fr": get_detokenizer(language="fr"),
}


def add_special_tokens(
    transcript_or_translation,
    source_lang,
    target_lang,
    tokenizer,
    include_xt=False,
    include_turn=False,
):
    """Function to construct the prompt to the model (ground truth) with special tokens
    - The format should be [source_lang] [target_lang] ... [special-tokens]
    """

    # if you degine the source and target langs with "[]", e.g., "[en]"
    source_lang = source_lang if "[" in source_lang else f"[{source_lang}]"
    source_lang = tokenizer.encode_as_ids(source_lang)[1]
    target_lang = target_lang if "[" in target_lang else f"[{target_lang}]"
    target_lang = tokenizer.encode_as_ids(target_lang)[1]

    # remove "[turn]" or ["xt"] tokens
    if not include_xt:
        transcript_or_translation.replace("[xt]", "")
    if not include_turn:
        transcript_or_translation.replace("[turn]", "")
    # get the tokens for text and join with source and target language
    tokens_list = tokenizer.encode_as_ids(transcript_or_translation)
    tokens_list = [source_lang, target_lang] + tokens_list

    return transcript_or_translation, tokens_list


def load_datasets(pipelines, hparams):
    """load the datasets based on the hparams passed to the function"""

    (
        audio_pipeline_train,
        audio_pipeline,
        one_reference_text_pipeline,
        four_reference_text_pipeline,
    ) = pipelines

    datasets = {}
    data_folder = hparams["data_folder"]
    train_dev = [hparams["train_splits"], hparams["dev_splits"]]

    # process train and dev dataset (only for each subset)
    for dataset in train_dev:
        json_path = f"{data_folder}/{dataset}.json"

        dataset = "train" if "train" in dataset else "valid"
        is_use_sp = "train" in dataset and "speed_perturb" in hparams
        audio_pipeline_func = audio_pipeline_train if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": FISHER_DATA_FOLDER},
            dynamic_items=[
                audio_pipeline_func,
                one_reference_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "task",
                "source_lang",
                "target_lang",
                "transcription",
                "translation_0",
                "translation_or_transcription",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )

    # process dev datasets with 4 transcriptions
    for dataset in hparams["test_splits_4_translations"]:
        json_path = f"{data_folder}/{dataset}.json"
        dataset = "_".join(dataset.split("/")[-2:])

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": FISHER_DATA_FOLDER},
            dynamic_items=[
                audio_pipeline,
                four_reference_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "task",
                "source_lang",
                "target_lang",
                "translation_or_transcription",
                "translation_0",
                "translation_1",
                "translation_2",
                "translation_3",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )

    # CALLHOME EVALUATION SETS, only one reference file
    for dataset in hparams["test_splits_1_translations"]:
        json_path = f"{data_folder}/{dataset}.json"
        dataset = "_".join(dataset.split("/")[-2:])

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": FISHER_DATA_FOLDER},
            dynamic_items=[
                audio_pipeline,
                one_reference_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "task",
                "source_lang",
                "target_lang",
                "transcription",
                "translation_0",
                "translation_or_transcription",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )
    return datasets


def sort_datasets(datasets, hparams):
    """sort & shuffle the datasets based on hparams"""

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(sort_key="duration")
        datasets["valid"] = datasets["valid"].filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration", reverse=True
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")
    return datasets


def load_dynamic_batcher(datasets, hparams):
    """Function to load the dynamic batcing objetcs.
    It will use datasets["train"] and datasets["valid"] objects
    """
    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

    dynamic_hparams = hparams["dynamic_batch_sampler"]
    num_buckets = dynamic_hparams["num_buckets"]

    train_bsampler = DynamicBatchSampler(
        datasets["train"],
        dynamic_hparams["max_batch_len"],
        num_buckets=num_buckets,
        length_func=lambda x: x["duration"],
        shuffle=dynamic_hparams["shuffle_ex"],
        batch_ordering=dynamic_hparams["batch_ordering"],
        max_batch_ex=dynamic_hparams["max_batch_ex"],
    )
    valid_bsampler = DynamicBatchSampler(
        datasets["valid"],
        dynamic_hparams["max_batch_len_val"],
        num_buckets=num_buckets,
        length_func=lambda x: x["duration"],
        shuffle=dynamic_hparams["shuffle_ex"],
        batch_ordering=dynamic_hparams["batch_ordering"],
    )

    return train_bsampler, valid_bsampler


def initialize_beam_search(beam_search_object, source_lang, target_lang, hparams):
    """Function to initialize beam Search objects.
    You need to pass the beam search object, the tokenizer and the source
    and target language.
    """
    # Get the target and source language IDS from the Tokenizer, langs are set as --> [xx]
    source_lang_id = hparams.tokenizer.encode_as_ids(f"[{source_lang}]")[-1]
    target_lang_id = hparams.tokenizer.encode_as_ids(f"[{target_lang}]")[-1]

    beam_search_object.set_decoder_prefix_tokens(
        source_lang=source_lang_id, target_lang=target_lang_id
    )


def print_bleu_or_wer(metrics, filepath, logger, is_bleu=False):
    """Simple function to print in a csv file the groun truth and outputs of the model
    We also print the default metrics of the model! with 'write_stats'
    """

    # first print the SpeechBrain standard stats in the given filepath
    with open(filepath, "w", encoding="utf-8") as w:
        metrics.write_stats(w)

    if is_bleu:
        csv_lines = [
            [id, target, predict]
            for id, target, predict in zip(
                metrics.ids, metrics.targets[0], metrics.predicts
            )
        ]
    # otherwise, it is an ASR transcription
    else:
        csv_lines = [
            [score["key"], " ".join(score["ref_tokens"]), " ".join(score["hyp_tokens"])]
            for score in metrics.scores
        ]

    # use the same filepath but change to CSV to save outputs!
    filepath = filepath.replace(".txt", ".csv")

    # insert the header
    csv_lines.insert(0, ["ID", "gt", "prediction"])
    # Writing the csv_lines
    with open(filepath, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully wrote the models' outputs!" % (filepath)
    logger.info(msg)


def print_inference_output(ids, ground_truth, predictions, filepath):
    """Simple function to print in a csv file the output of the model.
    We follow the convention of id;gt;prediction, thus gt will be empty
    """

    is_translation = True if "bleu_" in filepath else False

    assert len(ids) == len(predictions), "Nb. IDs does not match Nb. predictions"

    # the ground truth file is JSON
    with open(ground_truth, "r") as f:
        gt_data = json.load(f)

    # gather the ground truth and predictions in these dictionaries
    gt_dict, pred_dict = {}, {}

    # collect the output of the model here
    for utt_id, pred in zip(ids, predictions):
        utt_id = utt_id.split("-")[0]
        if utt_id not in pred_dict:
            pred_dict[utt_id] = f"{pred}"
            continue
        pred_dict[utt_id] = f"{pred_dict[utt_id]} [turn] {pred}"

    # collect the ground truth data in dictionary
    for utt_id, value in gt_data.items():
        utt_id = utt_id.split("-")[0]
        value = value["translation_0"] if is_translation else value["transcription"]
        if utt_id not in gt_dict:
            gt_dict[utt_id] = f"{value}"
            continue
        gt_dict[utt_id] = f"{gt_dict[utt_id]} [turn] {value}"

    # parse the id and predictions in one list
    csv_lines = [[utt_id, "", pred] for utt_id, pred in pred_dict.items()]

    # use the same filepath but change to CSV to save outputs!
    filepath = filepath.replace(".txt", ".csv")

    # insert the header
    csv_lines.insert(0, ["ID", "gt", "prediction"])
    # Writing the csv_lines
    with open(filepath, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # printing a new ground truth file
    csv_lines = [[utt_id, tgt, ""] for utt_id, tgt in gt_dict.items()]

    # create a file with the ground truth
    filepath = (
        filepath.replace(".txt", ".csv")
        .replace("-asr.csv", "-gt.csv")
        .replace("-st.csv", "-gt.csv")
    )

    # insert the header
    csv_lines.insert(0, ["ID", "gt", "prediction"])
    # Writing the csv_lines
    with open(filepath, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully wrote the models' outputs!" % (filepath)
    print(msg)


def append_gt_preds(
    ids,
    ref,
    hyps,
    target_lang,
    tokenizer,
    remove_special_chars=False,
    chars_dict=None,
):
    """Function to get the ground truth and outputs in lists!
    chars_dict: needs to be in this format:
    {
        "[turn]": 9
        "[xt]": 10
    },
        where the key is the word to remove from the transcript and the element
        is the position tokenizer ID
    """

    # assert there's at least one token to
    if remove_special_chars == True and chars_dict is None:
        print(
            f"chars_dict cannot be empty if remove_special_chars == {remove_special_chars}"
        )
        return None
    if remove_special_chars == True and not isinstance(chars_dict, dict):
        print("chars_dict needs to be a dict")
        return None

    # load the target detokenizer
    target_detokenizer = _DETOKENIZERS[target_lang]

    # lists to store predictions and targets of ASR/ST
    ids_list, ref_list, hyps_list = [], [], []
    # now iterate over the data and append the id/ref/hyps
    for id_utt, tgts, utt_seq in zip(ids, ref, hyps):
        # if remove_special_chars==True: clean tgts and hyps of 'turn and xt' information
        if remove_special_chars:
            for key, value in chars_dict.items():
                tgts = tgts.replace(key, "").replace("  ", " ")
                utt_seq = [i for i in utt_seq if i != value]

        # detokenize ref and hyp
        tgts = target_detokenizer.detokenize(tgts.split(" "))
        utt_seq = target_detokenizer.detokenize(
            tokenizer.decode_ids(utt_seq).split(" ")
        )

        # append ids and detokenized ref/hyp outputs
        ids_list.append(id_utt)
        ref_list.append(tgts)
        hyps_list.append(utt_seq)

    return ids_list, ref_list, hyps_list


def append_4gt(
    refs,
    target_lang,
    chars_dict,
):
    """Function to append in a list dev/test sets with 4 ground truth!
    chars_dict: needs to be in this format:
    {
        "[turn]": 9
        "[xt]": 10
    },
        where the key is the word to remove from the transcript and the element
        is the position tokenizer ID
    """

    # assert there's at least one token to
    if not isinstance(chars_dict, dict):
        print("chars_dict needs to be a dict!")
        return None

    # load the target detokenizer
    target_detokenizer = _DETOKENIZERS[target_lang]

    targets, targets_no_turn = [], []
    # iterate over the reference
    for reference in refs:
        detokenized_translation = [
            target_detokenizer.detokenize(translation.split(" "))
            for translation in reference
        ]
        targets.append(detokenized_translation)

        # append the references without the '[turn]' token
        for key, value in chars_dict.items():
            reference = [x.replace(key, "").replace("  ", " ") for x in reference]

        detokenized_translation = [
            target_detokenizer.detokenize(translation.split(" "))
            for translation in reference
        ]
        targets_no_turn.append(detokenized_translation)

    return targets, targets_no_turn
