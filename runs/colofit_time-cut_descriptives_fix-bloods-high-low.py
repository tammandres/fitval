"""Summarise blood test results on all data and by time period 
Fixes an issue where HGB counts were at times incorrect due to the F category being labelled 
as 'Female or non-binary (<10)', but the bloods_high_low() function only accepted M or F.
Also hides only the low counts itself, as the percentages do not sum to 100,
so there is no risk of low counts being inferred from the total.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
import dataclasses

os.chdir(r"E:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/")
from dataprep.coredata import load_coredata, load_additional_data
from dataprep.inclusion import _include_subj
from dataprep.summarise import summary_table_bloodsonly


PROJECT_ROOT = Path("E:/Users/5lJC/Desktop/fitval_may2024/")


# Load data
fu = 180
data_path = Path('E:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))

cdata = load_coredata(data_path)
adata = load_additional_data(data_path)

df = pd.read_csv(data_path / 'data_matrix.csv')
fit = pd.read_csv(data_path / 'fit.csv')
fit.fit_date = pd.to_datetime(fit.fit_date)

# Split into periods (NB - same as in metrics script)
precovid = fit.loc[fit.fit_date < '2020-03-01']
covid = fit.loc[(fit.fit_date >= '2020-03-01') & (fit.fit_date < '2021-05-01')]
post1 = fit.loc[(fit.fit_date >= '2021-05-01') & (fit.fit_date < '2022-01-01')]
post2 = fit.loc[(fit.fit_date >= '2022-01-01') & (fit.fit_date < '2022-07-01')]
post3 = fit.loc[(fit.fit_date >= '2022-07-01') & (fit.fit_date < '2023-01-01')]
post4 = fit.loc[(fit.fit_date >= '2023-01-01') & (fit.fit_date < '2023-07-01')]

date_thr = fit.loc[fit.stool_pot == 1].fit_date.min()
fit = fit.loc[(fit.stool_pot == 0) & (fit.fit_date >= date_thr)]

post3b = fit.loc[(fit.fit_date >= '2022-07-01') & (fit.fit_date < '2023-01-01')]
post4b = fit.loc[(fit.fit_date >= '2023-01-01') & (fit.fit_date < '2023-07-01')]
print(post3b.fit_date.min(), post3b.fit_date.max())
print(post4b.fit_date.min(), post4b.fit_date.max())

fit_data = {'precovid': precovid, 
            'covid': covid, 
            'post1': post1,
            'post2': post2, 
            'post3': post3,
            'post4': post4
            }

# Loop over data within time periods
stat = pd.DataFrame()
for label, f in fit_data.items():
    if label not in ['covid', 'post2', 'post3']:
        continue
    pat = f.patient_id.unique()
    print('\n---', label, len(pat))
    cdata2, adata2 = dataclasses.replace(cdata), dataclasses.replace(adata)
    cdata2 = _include_subj(cdata2, pat)
    print(cdata2.demo.gender.unique())
    adata2 = _include_subj(adata2, pat)
    print(cdata2.fit.shape, cdata.fit.shape)
    s = summary_table_bloodsonly(None, cdata=cdata2, adata=adata2)
    s = s.rename(columns={'No colorectal cancer': label + '_nocrc',
                          'Colorectal cancer': label + '_crc'})
    s = s.set_index(['characteristic', 'group'])
    stat = pd.concat(objs=[stat, s], axis=1)

stat = stat.reset_index()
stat = stat.sort_values(by='group')


# Only retain some blood test results 
# (HGB - to get new correct values, plus white cells and mch - to get counts that need not be hidden)
mask = stat.group.isin(['g02_',  'g04_', 'g08_', 'g10_', 'g11_', 'g13_'])
stat = stat.loc[mask]

save_dir = PROJECT_ROOT / 'results' / ('fixhgb_timecut_fu-' + str(fu))
save_dir.mkdir(exist_ok=True, parents=True)
stat.to_csv(save_dir / 'update_bloods_hilo_by_timeperiod.csv', index=False)


# All data
s = summary_table_bloodsonly(None, cdata=cdata, adata=adata)
mask = s.group.isin(['g04_'])
s = s.loc[mask]
s.to_csv(save_dir / 'update_bloods_hilo.csv', index=False)

