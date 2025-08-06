"""
From reviewer:
The first FIT value was selected for each patient – 
is information available on how many subsequent tests were excluded, 
and whether any of these led to a cancer diagnosis? 
I note 168 CRC were recorded more than 180 days after (the first) FIT, 
but these could have had a subsequent FIT. In practice, patients’ 
first test result may not be what leads to investigation, 
so it is important to understand where the analysis presented does not fully mirror clinical practice.
2025-05-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from dataprep.fit import _fit_clean, _fit_get


# Paths
raw_data_path = Path('C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/data_parquet')
data_path = Path('C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/data-colofit_fu-180')
data_before_inclusion = Path('C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/data_before_inclusion')

# Earliest diagnosis date for included patients (makes sense, as patients who had cancer before would not be included)
diagmin = pd.read_csv(data_before_inclusion / 'diagmin.csv')
diagmin.days_fit_to_diag.describe()
diagmin.diagnosis_source.value_counts()

# FIT values included in analysis (first FIT per patient, 180-day fu, no crc before FIT)
fit_incl = pd.read_csv(data_path / 'fit.csv')
fit_incl.fit_date = pd.to_datetime(fit_incl.fit_date)
print(fit_incl.shape, fit_incl.patient_id.nunique())
diagmin_incl = pd.read_csv(data_path / 'diagmin.csv')
diagmin_incl.days_fit_to_diag.describe()

# How many additional FITs could we include in analysis?
#fit = pd.read_parquet(raw_data_path / 'fit_values')
fit = _fit_get(raw_data_path)
print(fit.shape, fit.patient_id.nunique())

fit = _fit_clean(fit)
print(fit.shape, fit.patient_id.nunique())

fit = fit.loc[fit.gp.astype(int) == 1]
print(fit.shape, fit.patient_id.nunique())

fit = fit.loc[fit.patient_id.isin(fit_incl.patient_id)] # FITs for included patients
print(fit.shape, fit.patient_id.nunique())

fit = fit.loc[~fit.fit_val.isna()]  # FITs with FIT values
print(fit.shape, fit.patient_id.nunique())

fit = fit[['patient_id', 'fit_date', 'icen', 'fit_date_received', 'fit_val']].drop_duplicates()
print(fit.shape, fit.patient_id.nunique())

fit.shape[0] - fit_incl.shape[0]  # Roughly check potential gain in FITs, 15473

fit = fit.loc[~fit.patient_id.isin(diagmin_incl.patient_id)] # FITs not already associated with cancer
print(fit.shape, fit.patient_id.nunique())

# Correct FIT sample date
print('\nCorrecting FIT sample date...')
gpreqs = pd.read_parquet(raw_data_path / 'gpreqs')
fit = fit.merge(gpreqs[['patient_id', 'icen', 'fit_request_date', 'fit_request_or_sample_date']], how='left')
print(fit.shape, fit.patient_id.nunique())
fit['fit_request_date_corrected'] = fit.fit_request_or_sample_date.copy()
mask0 = ~fit.fit_request_date.isna()
fit.loc[mask0, 'fit_request_date_corrected'] = fit.loc[mask0, 'fit_request_date']
mask = fit.fit_request_date_corrected == fit.fit_date
print("In {} observations, FIT sample collection date was equal to request date, and was replaced with sample received date".format(mask.sum()))
fit.loc[mask, 'fit_date'] = fit.loc[mask, 'fit_date_received']
test = fit.fit_date == fit.fit_date_received
print("{} observations ({}%) have sample date unknown and was set equal to date received".format(test.sum(), test.mean()*100))

# Only retain patients with 2+ FITs on different dates
nfit = fit.groupby('patient_id').size()
mask = nfit > 1
nfit2 = nfit[mask]
print(mask.sum())

test = fit.groupby('patient_id').fit_date.nunique()
nfit2_difdate = test[test > 1]
print(nfit2_difdate.shape)

fit = fit.loc[fit.patient_id.isin(nfit2_difdate.index)]
print(fit.shape, fit.patient_id.nunique())

# Drop the first FIT
#first_fit_date = fit.groupby('patient_id').fit_date.min().rename('first_fit_date').reset_index()
#fit = fit.merge(first_fit_date, how='left')
fit = fit.merge(fit_incl[['patient_id', 'fit_date']].rename(columns={'fit_date': 'first_fit_date'}), how='left')
fit.first_fit_date.isna().sum()
print(fit.shape, fit.patient_id.nunique())

fit = fit.loc[fit.fit_date > fit.first_fit_date]
print(fit.shape, fit.patient_id.nunique())
assert fit.patient_id.isin(nfit2.index).all()
fit[['patient_id', 'fit_date']].merge(fit_incl[['patient_id', 'fit_date']], how='inner')

# Check num tests left: 15176
fit = fit[['patient_id', 'fit_date', 'first_fit_date', 'fit_val']]
print(fit.shape)
fit = fit.drop_duplicates()
print(fit.shape)

idx = fit.groupby(['patient_id', 'fit_date']).fit_val.idxmax()
fit = fit.loc[idx]
assert fit.shape[0] == fit[['patient_id', 'fit_date']].drop_duplicates().shape[0]
print(fit.shape[0])

# Check num of these assoc. with cancer
diagmin_new = diagmin[['patient_id', 'diagnosis_date', 'diagnosis_source']].merge(fit_incl[['patient_id', 'fit_date']], how='inner')
diagmin_new.diagnosis_date = pd.to_datetime(diagmin_new.diagnosis_date)
diagmin_new['delta'] = (diagmin_new.diagnosis_date - diagmin_new.fit_date).dt.days
print(diagmin_new.delta.describe())

diagmin_new = diagmin[['patient_id', 'diagnosis_date', 'diagnosis_source']].merge(fit, how='inner')
diagmin_new.diagnosis_date = pd.to_datetime(diagmin_new.diagnosis_date)
diagmin_new['delta'] = (diagmin_new.diagnosis_date - diagmin_new.fit_date).dt.days
print(diagmin_new.delta.describe())
print(diagmin_new.shape[0], diagmin_new.patient_id.nunique())

diagmin_new = diagmin_new.loc[(diagmin_new.delta >= 0) & (diagmin_new.delta <= 180)]
diagmin_new = diagmin_new.sort_values(by=['patient_id', 'fit_date'])
diagmin_new
print(diagmin_new.shape[0], diagmin_new.patient_id.nunique())

diagmin_new = diagmin_new.loc[diagmin_new.groupby('patient_id').fit_date.idxmin()]
print(diagmin_new.shape[0], diagmin_new.patient_id.nunique())
fit.shape

p = diagmin_new.patient_id.nunique() / fit.shape[0] * 100
print(p)


# Summarise TNM stage better
diagmin_incl.columns
ncrc = diagmin_incl.shape[0]
print(ncrc)
for col in ['T', 'N', 'M']:
    data = diagmin_incl[col].str.replace('[a-d]', '', regex=True).fillna('NULL')
    data = data.str.replace('[xX]', 'NULL', regex=True).sort_values()
    n = data.value_counts()
    p = np.round(n / ncrc * 100, 1)
    s = pd.concat(objs=[n, p], axis=1)
    s.columns = ['n', 'p']
    print(s)

diagmin_incl[['T', 'N', 'M']]
