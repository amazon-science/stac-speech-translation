#!/usr/bin/env/python3
"""Recipe for training a Transformer based ST system with Fisher-Callhome.
The system employs an encoder and a decoder. 

- We perform direct Speec-to-text translation. 
We add a CTC layer in the output of the decoder to help the alignment task.

Decoding is performed with beam search. 

To run this recipe, do the following:
> python train_multitask.py hparams/conformer.yaml


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
    add_special_tokens,
    append_4gt,
    append_gt_preds,
    get_detokenizer,
    initialize_beam_search,
    load_datasets,
    load_dynamic_batcher,
    print_bleu_or_wer,
    sort_datasets,
)
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

# deactivate numba logger
logging.getLogger("numba").setLevel(logging.WARNING)


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos  # for transcription/translation task

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # add data augmentation SpecAugment
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # ST outputs for CTC loss
        p_ctc = None
        if self.hparams.ctc_weight > 0:
            logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)

        # ST output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the ST is doing in general

                # our dev set contains transcripts and translation if number of tasks are 2
                if self.hparams.number_of_tasks >= 2:
                    # initialize beam search object for transcription
                    initialize_beam_search(
                        self.hparams.valid_search,
                        batch.source_lang[0],
                        batch.source_lang[0],
                        self.hparams,
                    )
                    hyps_asr, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
                    # initialize beam search object for translation
                    initialize_beam_search(
                        self.hparams.valid_search,
                        batch.source_lang[0],
                        batch.target_lang[0],
                        self.hparams,
                    )
                    hyps_st, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
                    hyps = [hyps_st, hyps_asr]  # pack the results
                else:
                    assert (
                        len(set(list(batch.task))) == 1
                    ), "number_of_tasks=1, but your val set has 1+ tasks"
                    # At test time, each JSON file represents one task, so we use source and target
                    initialize_beam_search(
                        self.hparams.valid_search,
                        batch.source_lang[0],
                        batch.target_lang[0],
                        self.hparams,
                    )
                    hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)

        # at test time, each dataset/JSON files, only does one task!
        elif stage == sb.Stage.TEST:
            # At test time, each JSON file represents one task, so we use source and target
            initialize_beam_search(
                self.hparams.test_search,
                batch.source_lang[0],
                batch.target_lang[0],
                self.hparams,
            )
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, p_seq, wav_lens, hyps) = predictions

        # get the batch data
        ids = batch.id
        tokens, tokens_lens = batch.tokens  # for CTC loss on translatation
        (
            tokens_eos,
            tokens_eos_lens,
        ) = batch.tokens_eos  # for ST translation, decoder output

        # loss for different tasks
        # loss = ctc_weight * ctc loss + (1 - ctc_weight) * NLL loss
        attention_loss = 0
        ctc_loss = 0

        # ST attention loss (NLL loss)
        attention_loss = self.hparams.seq_cost(
            p_seq,
            tokens_eos,
            length=tokens_eos_lens,
        )
        # CTC loss
        if self.hparams.ctc_weight > 0:
            ctc_loss = self.hparams.ctc_cost(
                p_ctc,
                tokens,
                wav_lens,
                tokens_lens,
            )

        # compute the actual loss
        loss = (
            self.hparams.ctc_weight * ctc_loss
            + (1 - self.hparams.ctc_weight) * attention_loss
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            # set the special tokens we would like to remove to get long-form audio ASR/ST clean scores
            special_tokens = {"[turn]": self.hparams.turn, "[xt]": self.hparams.xt}

            if (
                current_epoch % valid_search_interval == 0 and stage == sb.Stage.VALID
            ) or stage == sb.Stage.TEST:
                # If number of tasks >= 2 means that at valid time we do ASR and ST
                if self.hparams.number_of_tasks >= 2 and stage == sb.Stage.VALID:
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
                # with turn or no turn
                ids_st, ids_asr = [], []
                targets_st, predictions_st = [], []
                targets_asr, predictions_asr = [], []

                targets_st_no_turn, predictions_st_no_turn = [], []
                targets_asr_no_turn, predictions_asr_no_turn = [], []
                for task in all_tasks:
                    if task == "translation":
                        predicted_tokens = hyps_st
                        ground_truth_tokens = batch.translation_0
                    elif task == "transcription":
                        predicted_tokens = hyps_asr
                        ground_truth_tokens = batch.transcription

                    id_utt, tgts, utt_seq = append_gt_preds(
                        batch.id,
                        ground_truth_tokens,
                        predicted_tokens,
                        batch.target_lang[0],
                        hparams["tokenizer"],
                    )

                    # get predictions witout [turn] and [xt] tokens
                    _, tgts_no_turn, utt_seq_no_turn = append_gt_preds(
                        batch.id,
                        ground_truth_tokens,
                        predicted_tokens,
                        batch.target_lang[0],
                        hparams["tokenizer"],
                        remove_special_chars=True,
                        chars_dict=special_tokens,
                    )

                    # if not translation, then it is transcription:
                    for a, b, c, d, e in zip(
                        id_utt, tgts, utt_seq, tgts_no_turn, utt_seq_no_turn
                    ):
                        if task == "translation":
                            ids_st.append(a)
                            targets_st.append(b)
                            predictions_st.append(c)
                            targets_st_no_turn.append(d)
                            predictions_st_no_turn.append(e)
                        elif task == "transcription":
                            ids_asr.append(a)
                            targets_asr.append(b.split(" "))
                            predictions_asr.append(c.split(" "))
                            targets_asr_no_turn.append(d.split(" "))
                            predictions_asr_no_turn.append(e.split(" "))
                        else:
                            print(f"error, the task ({task}) is not defined...")

                if stage == sb.Stage.TEST:
                    # 4 references bleu score
                    try:
                        # check if there is more than one translation
                        _ = getattr(batch, "translation_1")

                        four_references = [
                            batch.translation_or_transcription,
                            batch.translation_1,
                            batch.translation_2,
                            batch.translation_3,
                        ]
                        # extract the targets (with and without turn and xt)
                        targets, targets_no_turn = append_4gt(
                            refs=four_references,
                            target_lang=batch.target_lang[0],
                            chars_dict=special_tokens,
                        )
                        # append to the metric objects
                        self.bleu_metric.append(ids_st, predictions_st, targets)
                        self.bleu_metric_no_turn.append(
                            ids_st, predictions_st_no_turn, targets_no_turn
                        )

                    # that means that is only 1 transcript:
                    except:
                        # Report both, WER and BLEU scores
                        # there is only one translation, dataset callhome or ASR outputs
                        # ipdb.set_trace()
                        # if "[de]" not in predictions_st or "[en]" not in predictions_st:
                        self.bleu_metric.append(ids_st, predictions_st, [targets_st])
                        self.wer_metric.append(ids_asr, predictions_asr, targets_asr)

                        self.bleu_metric_no_turn.append(
                            ids_st, predictions_st_no_turn, [targets_st_no_turn]
                        )
                        self.wer_metric_no_turn.append(
                            ids_asr, predictions_asr_no_turn, targets_asr_no_turn
                        )

                elif (
                    current_epoch % valid_search_interval == 0
                    and stage == sb.Stage.VALID
                ):
                    # Report both, WER and BLEU scores
                    # there is only one translation, dataset callhome or ASR outputs

                    self.bleu_metric.append(ids_st, predictions_st, [targets_st])
                    self.wer_metric.append(ids_asr, predictions_asr, targets_asr)

                    self.bleu_metric_no_turn.append(
                        ids_st, predictions_st_no_turn, [targets_st_no_turn]
                    )
                    self.wer_metric_no_turn.append(
                        ids_asr, predictions_asr_no_turn, targets_asr_no_turn
                    )

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(loss / self.grad_accumulation_factor).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                # if self.check_gradients(loss):
                if self.check_loss_isfinite(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)
                # self.hparams.lr_scheduler(self.optimizer)

        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                # if self.check_gradients(loss):
                if self.check_loss_isfinite(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.lr_scheduler(self.optimizer, self.optimizer_step)
                # self.hparams.lr_scheduler(self.optimizer)

        if sb.utils.distributed.if_main_process():
            if float(self.optimizer.param_groups[0]["lr"]) < 0:
                print("lr is less than 0!")
                ipdb.set_trace()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.bleu_metric_no_turn = self.hparams.bleu_computer()
            self.wer_metric_no_turn = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                if len(self.bleu_metric.ids) > 0:
                    stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
                    stage_stats["BLEU_no_turn"] = self.bleu_metric_no_turn.summarize(
                        "BLEU"
                    )

                if len(self.wer_metric.ids) > 0:
                    stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                    stage_stats["WER_no_turn"] = self.wer_metric_no_turn.summarize(
                        "error_rate"
                    )

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            lr = self.optimizer.param_groups[0]["lr"]
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            # check if there's actually outputs for ASR, and print outputs:
            if len(self.wer_metric.ids) > 0:
                print_bleu_or_wer(self.wer_metric, self.hparams.wer_file, logger)
                print_bleu_or_wer(
                    self.wer_metric_no_turn, self.hparams.wer_file_no_turn, logger
                )

            # check if there's actually outputs for ST, and print outputs:
            if len(self.bleu_metric.ids) > 0:
                print_bleu_or_wer(
                    self.bleu_metric, self.hparams.bleu_file, logger, is_bleu=True
                )
                print_bleu_or_wer(
                    self.bleu_metric_no_turn,
                    self.hparams.bleu_file_no_turn,
                    logger,
                    is_bleu=True,
                )

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
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
        return loss.detach()


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

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        # final tensor
        sig = torch.tensor([])
        # wav field from JSON file might contain several wav files, concatenate them!
        for signal in wav.replace("  ", " ").split():
            signal, _ = librosa.load(signal, sr=hparams["sample_rate"])
            sig = torch.cat([sig, torch.tensor(signal)])
        if hparams["speed_perturb"]:
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes(
        "transcription", "translation_0", "source_lang", "target_lang"
    )
    @sb.utils.data_pipeline.provides(
        "translation_or_transcription",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def one_reference_text_pipeline(
        transcription, translation, source_lang, target_lang
    ):
        """Processes the transcriptions or translations to generate proper labels"""

        # select the annotation to use, if source_lang == target_lang is ASR, otherwise ST
        text = transcription if source_lang == target_lang else translation

        # append special tokens if it's the case
        text, tokens_list = add_special_tokens(
            transcript_or_translation=text,
            source_lang=source_lang,
            target_lang=target_lang,
            tokenizer=hparams["tokenizer"],
            include_xt=hparams["use_xt_token"],
            include_turn=hparams["use_turn_token"],
        )
        yield text
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes(
        "source_lang",
        "target_lang",
        "translation_0",
        "translation_1",
        "translation_2",
        "translation_3",
    )
    @sb.utils.data_pipeline.provides(
        "translation_or_transcription",
        "translation_1",
        "translation_2",
        "translation_3",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def four_reference_text_pipeline(source_lang, target_lang, *translations):
        """Processes the transcriptions to generate proper labels"""
        yield translations[0]
        yield translations[1]
        yield translations[2]
        yield translations[3]
        text = translations[0]

        # append special tokens if it's the case
        text, tokens_list = add_special_tokens(
            transcript_or_translation=text,
            source_lang=source_lang,
            target_lang=target_lang,
            tokenizer=hparams["tokenizer"],
            include_xt=hparams["use_xt_token"],
            include_turn=hparams["use_turn_token"],
        )
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # 2. pack the pipelines, you need to pass them in that order!
    pipelines = (
        audio_pipeline_train,
        audio_pipeline,
        one_reference_text_pipeline,
        four_reference_text_pipeline,
    )

    # 3. load the datasets defined in hparams and using the aboe-instantiated pipelines
    datasets = load_datasets(pipelines, hparams)

    # 4. Sort the datasets
    datasets = sort_datasets(datasets, hparams)

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_bsampler, valid_bsampler = None, None
    if hparams["dynamic_batching"]:
        train_bsampler, valid_bsampler = load_dynamic_batcher(datasets, hparams)

    return (
        datasets,
        train_bsampler,
        valid_bsampler,
    )


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
    logger.info("Info: " + f" training for {hparams['number_of_epochs']} epochs")

    # transcription/translation tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # We can now directly create the datasets for training, valid, and test
    datasets, train_bsampler, valid_bsampler = dataio_prepare(hparams)

    # for data in datasets:
    #     for sample in datasets[data]:
    # if "5375-" in sample["id"]:
    # ipdb.set_trace()
    # print(sample)
    # hparams["epoch_counter"].current = 5000
    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {
            "batch_sampler": valid_bsampler,
            "num_workers": hparams["num_workers"],
        }

    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    if hparams["no_eval"] == True:
        logger.info(
            "Info: "
            + f" we are not doing evaluation, because this was the training round!"
        )
    else:
        # selecting the datasets to evaluate
        test_splits = [
            hparams["test_splits_4_translations"],
            hparams["test_splits_1_translations"],
        ]
        test_splits = [item for sublist in test_splits for item in sublist]
        for dataset in test_splits:
            dataset = "_".join(dataset.split("/")[-2:])

            st_brain.hparams.bleu_file = os.path.join(
                hparams["output_folder"], "bleu_{}.txt".format(dataset)
            )
            st_brain.hparams.bleu_file_no_turn = os.path.join(
                hparams["output_folder"], "bleu_{}_no_turn.txt".format(dataset)
            )
            st_brain.hparams.wer_file = os.path.join(
                hparams["output_folder"], "wer_{}.txt".format(dataset)
            )
            st_brain.hparams.wer_file_no_turn = os.path.join(
                hparams["output_folder"], "wer_{}_no_turn.txt".format(dataset)
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
