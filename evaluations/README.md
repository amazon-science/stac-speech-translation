# BENCHMARKING STAC-ST

Follow this README to benchmark different aspects of STAC-ST. We compare:

- 1. STAC-ST versus [Whisper](https://cdn.openai.com/papers/whisper.pdf)
- 2. STAC-ST equipped with Voice Activity Detection ([with SHAS](https://github.com/mt-upc/SHAS)) for long-form ST
- 3. STAC-ST on Speaker Change Detection versus [PyAnnote Toolkit](https://github.com/pyannote/pyannote-audio)


---
## Long-form ST: VAD + STAC-ST

A common practice for translating long-form audio files is to first segment them into smaller chunks based on voice activity detection (VAD). We compare STAC-ST with different segmentation approaches, including SHAS, WebRCT and our proposed multi-turn & multi-speaker utterances.

To run this, you need to first run VAD with SHAS and WebRCT on the Fisher-CALLHOME corpora:

- 1. Run [run_shas_segmentation.sh](../datasets/fisher_callhome/run_shas_segmentation.sh) script, to pre-segment the dev/test subsets.
- 2. Select a pretrained model, e.g., `"./exp/stact-st/<seed>"` (see steps in main README)
- 3. Run the system with:

```
# set the STAC-ST model
pretrained_model="./exp/stact-st/"

# AND finally, run the evaluation
bash vad_shas/run_inference.sh $pretrained_model
```

- You need to set the `data_folder` path in `run_inference.sh`.
- The output of the inference will be stored in `$pretrained_model/inference`

Note that the inference script outputs RTTM files for `[turn]` and  `[xt]` tokens. The last part of the inference is to evaluate speaker change detection with `vad_shas/eval_speaker_change.sh`. An RTTTM example is below:

```
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 0.890 0.04 <NA> <NA> SPK1 <NA> <NA>
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 0.930 0.04 <NA> <NA> SPK1 <NA> <NA>
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 1.610 0.04 <NA> <NA> SPK1 <NA> <NA>
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 1.650 0.04 <NA> <NA> SPK1 <NA> <NA>
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 3.770 0.04 <NA> <NA> SPK1 <NA> <NA>
SPEAKER 20051028_180633_356_fsp-0-000025-002732-st 1 5.690 0.04 <NA> <NA> SPK1 <NA> <NA>
...
```

All these steps are run in `vad_shas/run_inference.sh`, including evaluation.

---
## STAC-ST versus Whisper 

Given the lack of prior work on multi-turn & multi-speakers ST, we compare STAC-ST against a strong multi-task model, i.e., Whisper.


To evaluate whisper you need to run [run_inference_whisper.sh](whisper/run_inference_whisper.sh), with:


```
# You might want to set the paramers below, inside whisper/run_inference_whisper.sh

# data folder
# data_folder="/folder/to/datasets/fisher_callhome_spanish/data_processed"

# which dataset configuration run, this runs both, single and multi-turn 
# segmentations="single-turn multi-turn"

# Set the model sizes to run
# model_sizes="whisper-tiny whisper-base whisper-small whisper-medium"

# AND finally, run the evaluation
bash whisper/run_inference_whisper.sh
```

After that, results should be listed in `exp_folder=exp/whisper`. 

---
## Speaker Change Detection with PyAnnote


By leveraging available annotations in Fisher-CALLHOME test sets, we measure speaker change detection performance with three standard metrics: False Alarm Rate (FAR), Miss Detection Rate (MDR) and F1-score. We compare against a well-known speaker diarization pipeline from PyAnnote Toolkit. 

To compute these metrics, we first prepare Rich Transcription Time Marked (RTTM) files for each test set from the time-aligned `CTC [turn] spikes`. This is done in the step `Long-form ST: VAD + STAC-ST`. 

To evaluate PyAnnote on Fisher-CALLHOME corpora you need to run [pyannote/run_inference_pyannote.sh](pyannote/run_inference_pyannote.sh), by:


- 1. Get and authentication token from HuggingFace (`use_auth_token`) and add it into `AUTO_TOKEN` var in [pyannote/run_inference_pyannote.sh](pyannote/run_inference_pyannote.sh)
- 2. Run the system with:


```
# You might want to set the paramers below, inside pyannote/run_inference_pyannote.sh

# data folder
# data_folder="/folder/to/datasets/fisher_callhome_spanish/data_processed"

# Authentication token form HuggingFace
# AUTO_TOKEN="qwerty-qwerty-qwerty"

# Set the PyAnnote pipelines to run
# model_names="pyannote/segmentation pyannote/speaker-diarization@2.1"

# AND finally, run the evaluation
bash pyannote/run_inference_pyannote.sh
```

