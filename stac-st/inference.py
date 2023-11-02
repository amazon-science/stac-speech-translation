#!/usr/bin/env/python3
"""Recipe for training a Transformer based ST system with Fisher-Callhome.
The system employs an encoder and a decoder. 

- We perform direct Speec-to-text translation. 
We add a CTC layer in the output of the decoder to help the alignment task.

Decoding is performed with beam search. 

To run this recipe, do the following:
> python inference.py hparams/transformer_inference.yaml


Author
------
 * Juan Zuluaga-Gomez, 2023

"""
import csv
import logging
import os
import sys

import ipdb
import librosa
import speechbrain as sb
import torch

# some other functions from utils
from dataio_and_utils import (
    append_gt_preds,
    initialize_beam_search,
    print_bleu_or_wer,
    print_inference_output,
)
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

# deactivate numba logger
logging.getLogger("numba").setLevel(logging.WARNING)

# lists to print the RTTM files for turn detection!
turn_rttm, xt_rttm = [], []
# the output of the encoder is at this frequency, change this in case you change frontend
# check the value by doing: int(model_outputs.shape[1] / float(batch.duration[0]))
DOWNSAMPLING = 25
FISHER_DATA_FOLDER = (
    "/folder/to/datasets/fisher_callhome_spanish/data_processed/data"
)


def append_speaker_turns(batch, model_ctc_outputs):
    """input: object with batch elements and the output of the encoder"""

    # get the outputs that are TURN or XT token IDs
    model_ctc_outputs = model_ctc_outputs.argmax(-1)
    pred_turn = model_ctc_outputs == hparams["turn"]
    pred_xt = model_ctc_outputs == hparams["xt"]

    pred_turn = pred_turn.cpu().numpy().astype(int)
    pred_xt = pred_xt.cpu().numpy().astype(int)

    for sample_idx in range(len(model_ctc_outputs)):
        cnt = 0
        # fetch utterance id and absolute start/end of the recording
        utt_id = batch.id[sample_idx]
        abs_start = int(utt_id.split("-")[2]) / 100.0

        for turn_sample, xt_sample in zip(pred_turn[sample_idx], pred_xt[sample_idx]):
            # getting the start time by ID
            start = cnt * (1 / DOWNSAMPLING)
            # update the turn and xt list with this information
            if turn_sample == 1:
                turn_rttm.append(
                    f"SPEAKER {utt_id} 1 {abs_start + start:.3f} {(1/DOWNSAMPLING)} <NA> <NA> SPK1 <NA> <NA>"
                )
            if xt_sample == 1:
                xt_rttm.append(
                    f"SPEAKER {utt_id} 1 {abs_start + start:.3f} {(1/DOWNSAMPLING)} <NA> <NA> SPK1 <NA> <NA>"
                )
            # increase the counter
            cnt += 1


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # compute features
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # forward CNN and encoder
        src = self.modules.CNN(feats)
        enc_out = self.modules.Transformer.encode(src, wav_lens)

        # ST or ASR outputs for CTC loss, we use these outputs for
        # speaker change detection --> prepare RTTM files with this!
        p_ctc = None
        if self.hparams.ctc_weight > 0:
            logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)

        if self.hparams.get_rttm_files == True:
            append_speaker_turns(batch, p_ctc)

        # our dev set contains transcripts and translation if number of tasks are 2
        if self.hparams.number_of_tasks >= 2:
            # initialize beam search object for transcription
            initialize_beam_search(
                self.hparams.test_search,
                batch.source_lang[0],
                batch.source_lang[0],
                self.hparams,
            )
            hyps_asr, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
            # initialize beam search object for translation
            initialize_beam_search(
                self.hparams.test_search,
                batch.source_lang[0],
                batch.target_lang[0],
                self.hparams,
            )
            hyps_st, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
            hyps = [hyps_st, hyps_asr]  # pack the results
        else:
            assert (
                len(set(list(batch.task))) == 1
            ), "number_of_tasks=1, but your val set has 1+ tasks"
            # At test time, each JSON file represents one task, so we use source and target
            initialize_beam_search(
                self.hparams.test_search,
                batch.source_lang[0],
                batch.target_lang[0],
                self.hparams,
            )
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, wav_lens, hyps) = predictions

        # get the batch data
        ids = batch.id

        # set the special tokens we would like to remove to clean scores for ST/ASR
        special_tokens = {"[turn]": self.hparams.turn, "[xt]": self.hparams.xt}

        # If number of tasks >= 2 means that at valid time we do ASR and ST
        if self.hparams.number_of_tasks >= 2:
            hyps_st, hyps_asr = hyps
            all_tasks = ["translation", "transcription"]
        else:  # test time and we only have one task per valid/test set
            if batch.task[0] == "transcription":
                all_tasks = ["transcription"]
                hyps_asr = hyps
            elif batch.task[0] == "translation":
                all_tasks = ["translation"]
                hyps_st = hyps

        # get the predictions and store them in different list, either ASR or ST
        for task in all_tasks:
            if task == "translation":
                predicted_tokens = hyps_st
                ground_truth_tokens = batch.translation_0
                target_lang = batch.target_lang[0]
            elif task == "transcription":
                predicted_tokens = hyps_asr
                ground_truth_tokens = batch.transcription
                target_lang = batch.source_lang[0]

            # get predictions witout [turn] and [xt] tokens
            id_utt, _, utt_seq = append_gt_preds(
                batch.id,
                ground_truth_tokens,
                predicted_tokens,
                target_lang,
                hparams["tokenizer"],
                remove_special_chars=True,
                chars_dict=special_tokens,
            )

            # append the results in the object to print
            for a, b in zip(utt_seq, id_utt):
                if b not in self.hparams.ids_list:
                    self.hparams.ids_list.append(b)

                if task == "translation":
                    self.hparams.st_list.append(a)
                elif task == "transcription":
                    self.hparams.asr_list.append(a)
                else:
                    print(f"error, the task ({task}) is not defined...")
        return None

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TEST:
            # check if there's actually outputs for ASR, and print outputs:
            if len(self.hparams.asr_list) > 0:
                print_inference_output(
                    self.hparams.ids_list,
                    st_brain.hparams.ground_truth,
                    self.hparams.asr_list,
                    self.hparams.wer_file,
                )

            # check if there's actually outputs for ST, and print outputs:
            if len(self.hparams.st_list) > 0:
                print_inference_output(
                    self.hparams.ids_list,
                    st_brain.hparams.ground_truth,
                    self.hparams.st_list,
                    self.hparams.bleu_file,
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return torch.tensor([0])


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        # final tensor
        sig = torch.tensor([])
        # wav field from JSON file might contain several wav files, concatenate them!
        for signal in wav.replace("  ", " ").split():
            signal, _ = librosa.load(signal, sr=hparams["sample_rate"])
            sig = torch.cat([sig, torch.tensor(signal)])
        return sig

    # 3. load the inference datasets defined in hparams and using the aboe-instantiated pipelines
    datasets = {}
    data_files = hparams["inference_splits"]

    # process train and dev dataset (only for each subset)
    for dataset in data_files.split(" "):
        json_path = f"{dataset}.json"
        dataset_id = json_path.split("/")[-2]

        # cehck if we need to fix the json files paths
        replace_data_root = (
            {"data_root": FISHER_DATA_FOLDER}
            if "/data_processed/data/" in json_path
            else {}
        )

        # prepare the object
        datasets[dataset_id] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements=replace_data_root,
            dynamic_items=[
                audio_pipeline,
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
            ],
        )

    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # transcription/translation tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)
    # for data in datasets:
    #     for sample in datasets[data]:
    #         ipdb.set_trace()
    #         print(sample)

    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # decoding each by each
    for json_path in hparams["inference_splits"].split(" "):
        dataset = json_path.split("/")[-2]

        turn_rttm, xt_rttm = [], []
        # set the ground truth JSON file to compute WER!
        st_brain.hparams.ground_truth = os.path.join(
            os.path.dirname(json_path), "data.json"
        )
        st_brain.hparams.asr_list = []
        st_brain.hparams.st_list = []
        st_brain.hparams.ids_list = []

        st_brain.hparams.bleu_file = os.path.join(
            hparams["output_folder"], f"bleu_{dataset}-st.csv"
        )
        st_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], f"wer_{dataset}-asr.csv"
        )

        # if decoding happened, omit it
        if os.path.isfile(st_brain.hparams.bleu_file) or os.path.isfile(
            st_brain.hparams.wer_file
        ):
            print(f"File present, not decoding again: {st_brain.hparams.bleu_file}")
            continue

        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )

        # print the RTTM files for inference --> TURN and XT tokens
        for file_s in ["turn", "xt"]:
            # set file and data to print
            filepath = os.path.join(
                hparams["output_folder"], f"RTTM_{dataset}_{file_s}.csv"
            )
            csv_lines = turn_rttm if file_s == "turn" else xt_rttm
            # write data into file
            with open(filepath, mode="w") as fp:
                for line in csv_lines:
                    fp.write(f"{line}\n")
