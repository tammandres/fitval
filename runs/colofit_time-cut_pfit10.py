"""Compute proportion with fit >= 10"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
from constants import PROJECT_ROOT


# Run
res = pd.DataFrame()
for fu in [180, 365]:

    # Read data and divide to time periods
    data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))

    df = pd.read_csv(data_path / 'data_matrix.csv')
    fit = pd.read_csv(data_path / 'fit.csv')
    fit.fit_date = pd.to_datetime(fit.fit_date)

    precovid = fit.loc[fit.fit_date < '2020-03-01']
    covid = fit.loc[(fit.fit_date >= '2020-03-01') & (fit.fit_date < '2021-05-01')]
    post1 = fit.loc[(fit.fit_date >= '2021-05-01') & (fit.fit_date < '2022-01-01')]
    post2 = fit.loc[(fit.fit_date >= '2022-01-01') & (fit.fit_date < '2022-07-01')]
    post3 = fit.loc[(fit.fit_date >= '2022-07-01') & (fit.fit_date < '2023-01-01')]
    post4 = fit.loc[(fit.fit_date >= '2023-01-01') & (fit.fit_date < '2023-07-01')]

    date_thr = fit.loc[fit.stool_pot == 1].fit_date.min()
    bfit = fit.loc[(fit.stool_pot == 0) & (fit.fit_date >= date_thr)]
    post3buff = bfit.loc[(bfit.fit_date >= '2022-07-01') & (bfit.fit_date < '2023-01-01')]
    post4buff = bfit.loc[(bfit.fit_date >= '2023-01-01') & (bfit.fit_date < '2023-07-01')]

    fit_data = {'all': fit,
                'precovid': precovid, 
                'covid': covid, 
                'post1': post1,
                'post2': post2, 
                'post3': post3,
                'post4': post4,
                'post3-buffcomm': post3buff,
                'post4-buffcomm': post4buff
                }
    
    if fu == 365:  # too small n
        fit_data.pop('post4')
        fit_data.pop('post4-buffcomm')
    
    matrix = {}
    for label, f in fit_data.items():
        dfsub = df.loc[df.patient_id.isin(f.patient_id)]
        matrix[label] = dfsub
        print(label, dfsub.shape[0], dfsub.crc.sum())

    out = {}
    for label, d in matrix.items():
        pfit10 = round((d.fit_val >= 10).mean() * 100, 2)
        ncrc = d.crc.sum()
        npat = d.shape[0]
        pcrc = round(ncrc / npat * 100, 2)
        r = {'npat': npat, 'ncrc': ncrc, 'pcrc': pcrc, 'pfit10': pfit10}
        out[label] = r
    out = pd.DataFrame.from_dict(out, orient='index')
    out.index.name = 'period'
    out = out.reset_index()
    out['fu'] = fu
    res = pd.concat(objs=[res, out], axis=0)

save_path = PROJECT_ROOT / 'results' / 'timecut_periods_sum'
save_path.mkdir(exist_ok=True)
res.to_csv(save_path / 'timecut_periods_sum.csv', index=False)
