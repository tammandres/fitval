"""Prepare data items before applying inclusion criteria
This script needs to be run only once.
"""
from constants import PROJECT_ROOT, DATA_DIR
from dataprep.coredata import coredata, additional_data
from dataprep.summarise import summary_table
import sys
import shutil


# Paths and settings
data_path = DATA_DIR
run_path = PROJECT_ROOT / 'data_before_inclusion'

if run_path.exists():
    shutil.rmtree(run_path)

run_path.mkdir(exist_ok=True)

save_data = True  # Save data to disk
days_treatment_before_diag = 180  # Treatments are considered for this number of days before CRC diagnosis
testmode = False  # Set to True to see if code runs (runs it only on a subset of data)
npat_bloods = 1000  # Includes blood tests available for at least 1000 patients to reduce file size
days_before_fit_blood = 385  # Accommodates COLOFIT and FITML
days_after_fit_blood = 50 # Accommodates COLOFIT and FITML


# Get core data
coredata_log = run_path / 'coredata.log'
with open(coredata_log, 'w') as f:
    sys.stdout = f
    cdata = coredata(run_path, data_path, save_data=True, gp_only=False, testmode=testmode)
sys.stdout = sys.__stdout__  # Reset stdout to the console


# Get additional data
coredata_log = run_path / 'adata.log'
with open(coredata_log, 'w') as f:
    sys.stdout = f
    adata = additional_data(run_path, fit=cdata.fit, data_path=data_path, save_data=True, testmode=testmode, 
                            npat_bloods=npat_bloods, days_before_fit_blood=days_before_fit_blood,
                            days_after_fit_blood=days_after_fit_blood)
sys.stdout = sys.__stdout__  # Reset stdout to the console


# Summarise
summarise_log = run_path / 'summarise.log'
with open(summarise_log, 'w') as f:
    sys.stdout = f
    s = summary_table(run_path, cdata=cdata, adata=adata, blood_method='nearest', save_data=True)
sys.stdout = sys.__stdout__  # Reset stdout to the console
