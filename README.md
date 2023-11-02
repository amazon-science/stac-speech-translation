# Speaker-Turn Aware Conversational Speech Translation (STAC-ST)

This repository contains the code for the EMNLP 2023 paper [End-to-End Single-Channel Speaker-Turn Aware
Conversational Speech Translation](https://arxiv.org/abs/2311.00697) by Juan Zuluaga-Gomez, Zhaocheng Huang, Xing Niu, Rohit Paturi,
Sundararajan Srinavasan, Prashant Mathur, Brian Thompson, and Marcello Federico.

## Citation

```
@inproceedings{zuluaga-gomez-etal-2023-end,
    title = "End-to-End Single-Channel Speaker-Turn Aware
Conversational Speech Translation",
    author = "Zuluaga-Gomez, Juan  and
      Huang, Zhaocheng  and
      Niu, Xing  and
      Paturi, Rohit  and
      Srinavasan, Sundararajan  and
      Mathur, Prashant  and
      Thompson, Brian  and
      Federico, Marcello",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2311.00697",
}
```

## Setup your environment

### 1. Clone STAC-ST repository.

```
mkdir -p ~/stac_st/
cd ~/stac_st/
git clone /this/repo/
```

### 2. Create conda environment with PYTORCH 2.0 and SpeechBrain

```
cd ~/stac_st/
conda create -n stac_st python=3.11.3
conda activate stac_st
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain/
```

(we can use SpeechBrain v0.5.14, above versions may break the training at some point)

### 3. Install SpeechBrain's dependencies:

```
pip install -r ../stac-st/requeriments_file.txt
# then install SpeechBrain
pip install --editable .
```

### 4. Important!!

For some reason, multi-GPU training does not work out of the box in EC2 machines. *TO ALLOW THE SYSTEM TO DO DDP, YOU NEED TO DO FOLLOWING*

```
1. vim ~/code/speechbrain/speechbrain/core.py
2. Go to line 321 and add the following line (only last line!):


# local_rank = None
# if "local_rank" in run_opts:
#     local_rank = run_opts["local_rank"]
# else:
#     if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "":
#         local_rank = int(os.environ["LOCAL_RANK"])
#         run_opts["local_rank"] = local_rank %<------------- ADD THIS LINE --|

```

At this stage, you should be able to run SpeechBrain recipes with multiple GPUs

### 5. Additional modules

There are 2 aditional scripts/modules that do not come with SpeechBrain, but we provide in `stac-st/modules/`:

- `stac-st/modules/TransformerMultiTask.py`: script to instantiate a Multitask Transformer model. It is simlar to the `TransformerASR()` class of SpeechBrain.
- `stac-st/modules/mutitask_decoder.py`: This sequence-to-sequence decoder allows the addition of special tokens (in our case, language tokens) as prompts before starting the decoding phase. This is not part of SpeechBrain, but the script is based on the `S2SMultiTaskTransformerBeamSearch()` class. 

Our scripts already load these modules directly.

---
## Prepare the datasets


The dataset folder is in `datasets/*`. There is one folder per dataset. Feel free to modify them depending on your needs!


% # TODO: probably we need to make one script that prepares all datasets

All the datasets are prepared in JSON format, example below:

```
  "sp_2101-B-061126-061245-st": {
    "wav": "{data_root}/callhome-train/wav/sp_2101-B-061126-061245.wav",
    "source_lang": "es",
    "target_lang": "en",
    "duration": 1.19,
    "task": "translation",
    "transcription": "y hoy no es el cuatro",
    "translation_0": "and today is not the fourth",
    "transcription_and_translation": "y hoy no es el cuatro\nand today is not the fourth"
  },
```

---
### Fisher-CALLHOME

To prepare this dataset go to `datasets/fisher_callhome`. Here, you will find the preparation scripts, but first, download the data! 

#### 1. Download the data

To donwload the Fisher-CALLHOME corpora, you need to have access to LDC. Download the "speech" data and the "transcripts" (that is why there are two links).

Fisher: 
 - https://catalog.ldc.upenn.edu/LDC96T17
 - https://catalog.ldc.upenn.edu/LDC96S35

After the extraction, that folders needs to look like this:
`/folder/LDC2010T04/fisher_spa/data/speech/` and `/LDC2010T04/fisher_spa_tr/data/transcripts/`

CALLHOME
 - https://catalog.ldc.upenn.edu/LDC2010S01
 - https://catalog.ldc.upenn.edu/LDC2010T04

After the extraction, that folders needs to look like this:
`/folder/LDC96T17/ch_sp/callhome/spanish/speech/` and `/folder/LDC96T17/callhome_spanish_trans_970711/transcrp/`

When you have the folders structured as above. You can continue with preparation

#### 2. Preparation scripts

The data preparation scripts are in:

1. Data preparation (single-turn): `run_data_preparation.sh`
2. Data preparation (multi-turn): `run_data_preparation_turns.sh`
3. Segmentation with SHAS and WebRCT: `run_shas_segmentation.sh`

For single-turn and multi-turn preparation run:
```
# first, open each file and set:
# ORIGINAL_DATA=/path/to/fisher/folder/
bash run_data_preparation.sh
bash run_data_preparation_turns.sh
```

For segmentation with SHAS and WebRCT:

```
# Open the script run_shas_segmentation.sh and set these two
fisher_data="/path/to/fisher/folder/"
callhome_data="/path/to/callhome/folder/"
# then
bash run_shas_segmentation.sh
```

At this point, you should have a tree like this:

```
.
├── audio_segmenter/
├── create_json_and_segment.py
├── data/
├── extra_requirements.txt
├── mask_wav_files.py
├── run_data_preparation.sh
├── run_data_preparation_turns.sh
├── run_shas_segmentation.sh
└── st_asr_task
```

---
### CommonVoice (CV)

To prepare this dataset go to `datasets/common_voice_13`. Here, you will find the preparation scripts.
The following command downloads and prepare the CommonVoice data!

```
# YOU NEED TO ENTER THESE PATHS!
# data_folder="/path/to/folder/datasets/common_voice_13/cv-corpus-13.0-2023-03-09"
# save_folder="/path/to/folder/datasets/common_voice_13/data"

# then run:
bash run_prepare_commonvoice.sh
```

---
### CoVoST2

To prepare this dataset go to `datasets/covost`. Here, you will find the preparation scripts.
The following command downloads and prepare the covost2 data!
**IMPORTANT, YOU NEED TO FIRST PREPARE COMMONVOICE AS COVOST2 DEPENDS ON IT**

```
# YOU NEED TO ENTER THESE PATHS!
# common_voice="/path/to/folder/datasets/common_voice_13/cv-corpus-13.0-2023-03-09"
covost_data="/path/to/folder/datasets/covost/splits"


# then run:
bash run_prepare_covost2.sh
```


---
### MSLT

To prepare this dataset go to `datasets/mslt`. Here, you will find the preparation scripts.
The following command downloads and prepare the MSLT data!

```
# YOU NEED TO ENTER THESE PATHS!
# data_folder="/path/to/folder/datasets/mslt"
# save_folder="/path/to/folder/datasets/mslt/data"

# then run:
bash run_prepare_mslt.sh
```


---
### CV + CoVoST2

A key part of STAC-ST is that it can use ASR or ST corpora from different datasets. Thus, in this step we merge in one JSON file the ASR and ST datasets i.e., CommonVoice + CoVoST2.


```
# YOU NEED TO SET THESE PATHS!
# datasets_folder=/path/to/folder/datasets
# mslt_dataset=${datasets_folder}/mslt/data/v_1/
# commonvoice=${datasets_folder}/common_voice_13/data/
# covost=${datasets_folder}/covost/splits/
# fisher_callhome=${datasets_folder}/fisher_callhome_spanish/data_processed/data


# then run:
bash prepare_cross_datasets.sh
```

The `prepare_cross_datasets.sh` prepares:

- CV train/dev/test single- and multi-turn subsets
- CoVoST2 train/dev/test single- and multi-turn subsets
- merges all data, single- and multi-turn, CV and CommonVoice.


---
## Training STAC-ST

When the datasets are ready, we can start training. 

Run the default model with:

```
bash run_default.sh
```
This trains a 5K BPE Tokenizer (`exp/tokenizer_bpe_5k_es_en`) and a STAC-ST model (`exp/stact-st/`). We train with single- and multi-turn Fisher-CALLHOME corpora.

Otherwise, you can run directly the following steps:

### 1. Train tokenizer

To train the tokenizer:

```
DATA_FOLDER="/path/to/dataset/fisher_callhome/data" # needs to be the folder that contains each subset
TOKENIZER="/path/to/tokenizer/"
lang_tokens=$(echo "[ES],[EN]")

python3 stac-st/train_tokenizer.py stac-st/hparams/train_bpe_5k_special_prefix.xx_to_xx.yaml \
    --languages "'$lang_tokens'" \
    --annotation_read "transcription_and_translation" \
    --train_json_file "$DATA_FOLDER/fisher-callhome-train/data-asr-st.json" \
    --output_folder "$TOKENIZER"
```

### 2. Train STAC-ST with Fisher-CALLHOME corpora

Train the model with:

```
DATA_FOLDER="/path/to/dataset/fisher_callhome/data # needs to be the folder that contains each subset"
TOKENIZER="/path/to/tokenizer/"
_TRAIN_SCRIPT="stac-st/train_multitask.py"
_hparams="stac-st/hparams/transformer_multitask.yaml"

# train split
_TRAIN_SPLIT="fisher-callhome-train-and-30s/data-turns-asr-st"

OMP_NUM_THREADS=40 torchrun \
    --nproc_per_node=$n_jobs \
    --nnodes=1 --node_rank=0 \
    ${_TRAIN_SCRIPT} ${_hparams} \
        --distributed_launch --distributed_backend='nccl' \
        --use_xt_token=True --use_turn_token=True \
        --encoder_module=transformer \
        --attention_type=regularMHA \
        --train_splits=$_TRAIN_SPLIT \
        --output_folder_name="./exp/stact-st/" \
        --max_batch_len=500 \
        --max_batch_len_val=250 \
        --test_batch_size=1 \
        --lr_adam=0.001 \
        --transformer_dropout=0.1 \
        --grad_accumulation_factor=8 \
        --scheduler_step_limit=100000 \
        --n_warmup_steps=10000 \
        --cooldown=10000 \
        --ctc_weight=0.3 \
        --d_model=256 \
        --d_ffn=1024 \
        --nhead=4 \
        --num_encoder_layers=12 \
        --num_decoder_layers=6 \
        --valid_search_interval=100 \
        --data_folder "$DATA_FOLDER" \
        --tokenizer_file $TOKENIZER/${units}_bpe.model 
```

Training parameters, note that max_batch_len tells how many seconds of audio each batch has 500 == 500s == 8.33mins | i.e, max_batch_len=500. 

The effective batch size is max_batch_len x grad_accumulation_factor, STAC-ST uses 500 x 8 (GPUs) == 4000s | i.e., grad_accumulation_factor=8


### 3. Train STAC-ST with Fisher-CALLHOME + CoVoST2 + CommonVoice

To train a model with out-of-domain corpora  (CoVoST2 + CommonVoice), you need to modify the data paths and the hyperparameters scripts. First, you need to prepare CoVoST2 and CommonVoice corpora and then run `datasets/cross_datasets_experiments/prepare_cross_datasets`.


For Tokenizer:

```
DATA_FOLDER="/path/to/dataset/cross_datasets_experiments/data_covost_mslt" 
TOKENIZER="/path/to/tokenizer/with/cv-covost2"

... (remains the same)

```


For STAC-ST model:

```
DATA_FOLDER="/path/to/dataset/cross_datasets_experiments/data_covost_mslt"
TOKENIZER="/path/to/tokenizer/with/cv-covost2"
_TRAIN_SCRIPT="stac-st/train_multitask.py"
_hparams="stac-st/hparams/transformer_fisher_cv_xx_to_xx.yaml"

# define source and target locale (languages)
source_locale="es"
target_locale="en"
lang_pair_id="${source_locale}_to_${source_locale}.and.${target_locale}"

# train split (contains single- and multi-turn data)
# The last part of the path can be changed, check the folder in $DATA_FOLDER/${lang_pair_id}.
# We have prepared data with CoVoST2 data only, single-turn data only, for CV+CoVoST2. 
# You can use different data configurations 

TRAIN_SPLIT="${lang_pair_id}/train-and-30s.${source_locale}_${target_locale}"


... (remains the same)

```

## Evaluate one STAC-ST model

You can evaluate a model by simply running the following code:


```
DATA_FOLDER="/path/to/dataset/fisher_callhome/data # needs to be the folder that contains each subset"
TOKENIZER="/path/to/tokenizer/"

_TRAIN_SCRIPT="stac-st/train_multitask.py"
_hparams="stac-st/hparams/transformer_fisher_cv_xx_to_xx.yaml"

# no_eval=False indicates to perform evaluation, (which is True by default, above)
# you need to set use_xt_token/use_turn_token to True/False, depending on what you set while training the model. It should match!

CUDA_VISIBLE_DEVICES=0 python3 \
  ${_TRAIN_SCRIPT} ${_hparams} \
    --no_eval=False --num_workers=12 \
    --use_xt_token=True --use_turn_token=True \
    --encoder_module=transformer \
    --attention_type=regularMHA \
    --train_splits=$_TRAIN_SPLIT \
    --output_folder_name="./exp/stact-st/" \
    --max_batch_len=100 \
    --max_batch_len_val=150 \
    --test_batch_size=1 \
    --lr_adam=0.001 \
    --transformer_dropout=0.1 \
    --grad_accumulation_factor=1 \
    --scheduler_step_limit=100000 \
    --n_warmup_steps=10000 \
    --cooldown=10000 \
    --ctc_weight=0.3 \
    --d_model=256 \
    --d_ffn=1024 \
    --nhead=4 \
    --num_encoder_layers=12 \
    --num_decoder_layers=6 \
    --valid_search_interval=100 \
    --data_folder=$DATA_FOLDER \
    --tokenizer_file=$TOKENIZER/${units}_bpe.model 
```

Note that this needs to be run after the training has been completed. Defined by `scheduler_step_limit`.

*IMPORTANT*: in `_hparams="stac-st/hparams/transformer_fisher_cv_xx_to_xx.yaml"`, the evaluation sets are defined as:

```
test_splits_1_translations: [
    !ref "<join>/covost_<source_locale>_<target_locale>_test-30s",
    !ref "<join>/covost_<source_locale>_<target_locale>_dev-30s",
    ...
]
```

You can define your own evaluation datasets as: 

```
test_splits_1_translations: [
    !ref "<join>/covost_<source_locale>_<target_locale>_test-30s",
    ...
    !ref "/path/to/eval/dataset/my_dummy_eval_set",
    ...
]
```

---
## Scaling up STAC-ST

Our model is scaled to different sizes, you can do this by running:

```
# small size (default)
bash ablations/run_m_and_l_size.sh "small"

# large size
bash ablations/run_m_and_l_size.sh "medium"

# XL size
bash ablations/run_m_and_l_size.sh "large"
```
Note that this scrips uses CommonVoice and CoVoST2 corpora for training.

---
## Other ablations

Here we list some details on how to perform the other ablations listed in the paper.

### Task Tokens ablations

To use/remove the special tokens: [turn] and [xt], we have added two flags to the training scripts:

```
--use_turn_token=True/False

and
--use_xt_token=True/False
```

These two flags can be set to True or False. 

### Setting CTC weight

CTC weight can be passed while training by setting

```
--ctc_weight=x.x

# where x.x goes from --> [0, 1)
```

### Training in other language directions

You need to change:

- When training Tokenizer, change `lang_tokens=$(echo "[ES],[EN]")` to `lang_tokens=$(echo "[DE],[EN]")` in case you want to do German-to-English ASR+ST.
- Set `source_locale="de"` and `target_locale="en"` when training STAC-ST. 



For step #2, see more in section: `3. Train STAC-ST with Fisher-CALLHOME + CoVoST2 + CommonVoice`


### Benchmarking STAC-ST

We benchmark STAC-ST in the following folder: [evaluations/README.md](evaluations/README.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.