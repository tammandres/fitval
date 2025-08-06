"""Compute summary statistics for time-cut (pre-cov, cov, post-cov, post1, ..., post4) datasets

NB before running this script, do

conda deactivate
conda activate fit-dataprep
cd "C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/"
python

Then copy code to terminal.
Alternatively, open the dataprep_fitml_and_fitval in VS code first.

Not ideal as uses code from different repo that is not a requirement of this repo.
Could move this code to the other repo?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
from dataprep.coredata import load_coredata, load_additional_data
from dataprep.summarise import summary_table
from dataprep.inclusion import _include_subj
import dataclasses


PROJECT_ROOT = Path("C:/Users/5lJC\Desktop/fitval_may2024/")


for fu in [180, 365]:
    print('Timecut analysis, fu', fu)

    # Load data with fu 365
    data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))

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
                'post4': post4,
                'post3-buffcomm': post3b,
                'post4-buffcomm': post4b
                }

    # Loop over data within time periods
    for label, f in fit_data.items():

        if fu == 365 and label.startswith('post4'):
            continue
        
        pat = f.patient_id.unique()

        # Destination dir - must already exist from running the metrics script
        out_name = 'impute-none_fu-' + str(fu) + '_period-' + label
        save_dir = PROJECT_ROOT / 'results' / ('timecut_fu-' + str(fu))
        save_path = save_dir / out_name
        if not save_dir.exists():
            raise ValueError("save_dir does not exist")

        print('\n---', label, len(pat), save_dir)
        cdata2, adata2 = dataclasses.replace(cdata), dataclasses.replace(adata)
        cdata2 = _include_subj(cdata2, pat)
        adata2 = _include_subj(adata2, pat)
        print(cdata2.fit.shape, cdata.fit.shape)

        # To avoid data loss in gender 
        u = cdata2.demo.gender.value_counts()
        print('Gender values', u.index)
        if 'I' in u.index and u['I'] < 10:
            mask = cdata.demo.gender.isin(['F', 'I'])
            cdata2.demo.loc[mask, 'gender'] = 'Female or non-binary (<10)'
        
        # Summary table by cancer
        s = summary_table(save_path, cdata=cdata2, adata=adata2, blood_method='nearest', save_data=True)
