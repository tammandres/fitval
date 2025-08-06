"""Apply inclusion criteria for FITML analyses"""
from constants import PROJECT_ROOT, DATACUT
from dataprep.inclusion import inclusion
from dataprep.matrix import matrix
from dataprep.summarise import summary_table
import sys


# ---- 365-day followup, all data ----
# Including bloods available for at least 90.0% of patients, equivalent to 36695 patients
#region
data_path = PROJECT_ROOT / 'data_before_inclusion'
run_path = PROJECT_ROOT / 'data-colofit_fu-365'
run_path.mkdir(exist_ok=True)

## Apply inclusion crit
inclusion_log = run_path / 'inclusion.log'
with open(inclusion_log, 'w') as f:
    sys.stdout = f
    cdata, adata, flow = inclusion(data_path=data_path, run_path=run_path,
                                days_followup=365, datacut=DATACUT, sym_only=False,
                                save_data=True, days_before_fit_blood=365,
                                days_after_fit_blood=14, days_before_fit_event=365,
                                days_after_fit_event=0, ppat_blood=0.9, npat_blood=None,
                                crc_before_fu=False, coreblood=False, coreblood_colofit=True, buffer='none',
                                npat_code=100)
sys.stdout = sys.__stdout__  # Reset stdout to the console
print(flow)
flow.to_csv(run_path / 'inclusion.csv', index=False)

## Summarise
summarise_log = run_path / 'summarise.log'
with open(summarise_log, 'w') as f:
    sys.stdout = f
    s = summary_table(run_path, cdata=cdata, adata=adata, blood_method='nearest', save_data=True)
sys.stdout = sys.__stdout__  # Reset stdout to the console

## Transform to data matrix
matrix_log = run_path / 'matrix.log'
with open(matrix_log, 'w') as f:
    sys.stdout = f
    df = matrix(run_path=run_path, save_data=True, blood_method='nearest', 
                ind_blood_hilo=0, codes=0, mis=True, incl_sym=0, incl_bmi=0, 
                incl_ind_blood=0, incl_eth=0, incl_imd=0, incl_bloods=1)
sys.stdout = sys.__stdout__  # Reset stdout to the console

## Explore performance of FIT
sens_fit = df.loc[df.fit_val >= 10].crc.sum() / df.crc.sum()
ppv_fit = df.loc[df.fit_val >= 10].crc.mean()
fit_pos = (df.fit_val >= 10).mean()
print(sens_fit, ppv_fit, fit_pos)

#endregion


# ---- 180-day followup, all data ----
#region
data_path = PROJECT_ROOT / 'data_before_inclusion'
run_path = PROJECT_ROOT / 'data-colofit_fu-180'
run_path.mkdir(exist_ok=True)

## Apply inclusion crit
inclusion_log = run_path / 'inclusion.log'
with open(inclusion_log, 'w') as f:
    sys.stdout = f
    cdata, adata, flow = inclusion(data_path=data_path, run_path=run_path,
                                days_followup=180, datacut=DATACUT, sym_only=False,
                                save_data=True, days_before_fit_blood=365,
                                days_after_fit_blood=14, days_before_fit_event=365,
                                days_after_fit_event=0, ppat_blood=0.9, npat_blood=None,
                                crc_before_fu=False, coreblood=False, coreblood_colofit=True, buffer='none',
                                npat_code=100)
sys.stdout = sys.__stdout__  # Reset stdout to the console
print(flow)
flow.to_csv(run_path / 'inclusion.csv', index=False)

## Summarise
summarise_log = run_path / 'summarise.log'
with open(summarise_log, 'w') as f:
    sys.stdout = f
    s = summary_table(run_path, cdata=cdata, adata=adata, blood_method='nearest', save_data=True)
sys.stdout = sys.__stdout__  # Reset stdout to the console

## Transform to data matrix
matrix_log = run_path / 'matrix.log'
with open(matrix_log, 'w') as f:
    sys.stdout = f
    df = matrix(run_path=run_path, save_data=True, blood_method='nearest', 
                ind_blood_hilo=0, codes=0, mis=True, incl_sym=0, incl_bmi=0, 
                incl_ind_blood=0, incl_eth=0, incl_imd=0, incl_bloods=1)
sys.stdout = sys.__stdout__  # Reset stdout to the console

## Explore performance of FIT
sens_fit = df.loc[df.fit_val >= 10].crc.sum() / df.crc.sum()
ppv_fit = df.loc[df.fit_val >= 10].crc.mean()
fit_pos = (df.fit_val >= 10).mean()
print(sens_fit, ppv_fit, fit_pos)

#endregion
