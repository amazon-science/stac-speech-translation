"""
Run the data preparation
"""
import sys

from st_asr_task.callhome_prepare_turns import prepare_turns_only_callhome_spanish
from st_asr_task.fisher_callhome_prepare_turns import (
    prepare_turns_fisher_callhome_spanish,
)

data_folder = sys.argv[1]
save_folder = sys.argv[2]
device = "cpu"

# let's prepare several versions of each dataset with "max time allowed"
# not that you increase that nubmer, so segments will be of longer duration

max_time_allowed = [30, 60]

for max_time in max_time_allowed:
    # preparing the ONLY callhome dataset
    prepare_turns_only_callhome_spanish(
        data_folder=data_folder,
        save_folder=save_folder,
        max_utterance_allowed=max_time,
        save_suffix="data-turns",
        device=device,
    )
    # preparing the fisher-callhome dataset
    prepare_turns_fisher_callhome_spanish(
        data_folder=data_folder,
        save_folder=save_folder,
        max_utterance_allowed=max_time,
        save_suffix="data-turns",
        device=device,
    )
