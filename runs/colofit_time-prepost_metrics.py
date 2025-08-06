"""Time period analysis, complete cases
Roughly runs 60 + 3 * 30 min = 2.5 hours"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
from constants import PROJECT_ROOT
from fitval.boot import boot_metrics


# Settings
B = 1000
M = 1
thr_risk = np.array([0.5, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]) / 100
thr_risk = list(thr_risk)
sens = [0.8, 0.85, 0.9, 0.95, 0.99]
thr_fit = [2, 10, 100]

model_names = ['nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot', 
               #'nottingham-fit', 'nottingham-fit-age', 
               'nottingham-fit-age-sex', 
               'nottingham-lr-3.5', 'nottingham-lr-quant',
               'nottingham-cox-3.5', 'nottingham-cox-quant',
               #'nottingham-fit-3.5', 'nottingham-fit-quant',
               #'nottingham-fit-age-3.5', 'nottingham-fit-age-quant',
               'nottingham-fit-age-sex-3.5', 'nottingham-fit-age-sex-quant',
               'fit'
               ]

# Run
for fu in [180, 365]:

    # Read data and divide to time periods
    data_path = Path('C:\\Users\\5lJC\Desktop\\dataprep_fitml_and_fitval\\data-colofit_fu-' + str(fu))
    df = pd.read_csv(data_path / 'data_matrix.csv')
    fit = pd.read_csv(data_path / 'fit.csv')
    fit.fit_date = pd.to_datetime(fit.fit_date)

    # Define data subsets for different periods
    date_thr = fit.loc[fit.stool_pot == 1].fit_date.min()
    buffcomm = fit.loc[(fit.stool_pot == 0) & (fit.fit_date >= date_thr)]
    bufftime = fit.loc[fit.fit_date > "2021-10-01"]
    prebuff = fit.loc[fit.fit_date < '2021-07-01']

    fit_data = {'all': fit,
                'buffcomm': buffcomm,
                'bufftime': bufftime,
                'prebuff': prebuff
                }

    matrix = {}
    for label, f in fit_data.items():
        dfsub = df.loc[df.patient_id.isin(f.patient_id)]
        matrix[label] = dfsub
        print(label, dfsub.shape[0], dfsub.crc.sum())

    # Run 
    for label, dfsub in matrix.items():
        print(label, dfsub.shape)

        cols = ['fit_val', 'age_at_fit', 'ind_gender_M', 'blood_PLT', 'blood_MCV']
        x = dfsub[cols]
        y = dfsub[['crc']]

        out_name = 'impute-none_fu-' + str(fu) + '_period-' + label
        save_path = PROJECT_ROOT / 'results' / ('prepost_fu-' + str(fu)) / out_name
        save_path.mkdir(exist_ok=True, parents=True)
        print(save_path)

        log = save_path / 'boot_metrics.log'
        with open(log, 'w') as f:
            sys.stdout = f
            data_ci, data_noci = boot_metrics(x, y, model_names=model_names, thr_risk=thr_risk, thr_fit=thr_fit,
                                            sens=sens, 
                                            B=B, M=M,
                                            save_path=save_path, save=True, parallel=True)
        sys.stdout = sys.__stdout__  # Reset stdout to the console
