"""
Run the data preparation
"""
import sys

from st_asr_task.callhome_prepare import prepare_only_callhome_spanish
from st_asr_task.fisher_callhome_prepare import prepare_fisher_callhome_spanish

data_folder = sys.argv[1]
save_folder = sys.argv[2]
device = "cpu"

# preparing the fisher-callhome dataset
prepare_fisher_callhome_spanish(data_folder, save_folder, device=device)

# preparing the callhome dataset
prepare_only_callhome_spanish(data_folder, save_folder, device=device)
