"""Reformat summary table, 2024-07-10"""
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from constants import PROJECT_ROOT


# Helper function
def second_reformat_disc_cal(data_path, pre, models, periods, time_label):

    # .... 1. Sensitivity at FIT thresholds ....
    #region
    fname = 'reformat_sens_fit.csv'

    # Read data, filter models and periods
    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]

    # Compute num patients and cancers for each time period
    def _float(x):
        x = re.sub(" *\(.*", "", x)
        x = float(x)
        return x

    df['Num patients'] = df['Positive tests'].apply(_float) + df['Negative tests'].apply(_float)
    df['Num cancers'] = df['Detected cancers'].apply(_float) + df['Missed cancers'].apply(_float)
    df['Prevalence (%)'] = (df['Num cancers'] / df['Num patients'] * 100).round(2)
    df_num = df[['Num patients', 'Num cancers', 'Prevalence (%)', 'period']].drop_duplicates().reset_index(drop=True)

    # Get sens of FIT >= 10 for fixing the subsequent metric_at_sens_table
    df_fit = df.loc[(df.Model == "FIT test") & (df['FIT threshold (ug/g)'] == 10)].copy()
    df_fit['sens'] = df_fit['Sensitivity (%)'].copy().str.replace(" *\(.*", "", regex=True).astype(float)
    df_fit = df_fit[['period', 'sens']].drop_duplicates()

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    first_cols = ['FIT threshold (ug/g)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                  'Positive tests per 1000 tests', 'Negative tests per 1000 tests', 
                  'Detected cancers per 1000 tests', 'Missed cancers per 1000 tests',
                  'False positive tests per 1000 tests', 'True negative tests per 1000 tests',
                  'Threshold approx (%)']
    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]
    
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)
    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())

    df_fit10 = df.loc[(df.Model == 'FIT test') & (df['FIT threshold (ug/g)'] == 10)]
    #endregion

    # .... 2. Gain in metric compared to FIT ....
    #region
    fname = 'reformat_sens_fit_gain2.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]

    # Dropping rows where the FIT test was not applied at original threshold, but at same sensitivity as original FIT
    df = df.loc[df.Model != 'FIT test']  
    df = df.merge(df_num, how='left', on='period')

    # Dropping rows where the FIT test was not applied at original threshold, but at same sensitivity as original FIT
    df = df.loc[df.Model != 'FIT test']  

    #  Get data with CI method 1 as well
    #df2 = pd.read_csv(data_path / 'reformat_sens_fit_gain.csv')
    #df2 = df2.loc[df2.Model.isin(models)]
    #df2 = df2.loc[df2.period.isin(periods)]
    #df2 = df2.loc[df2.Model != 'FIT test'] 
    #df2 = df2.rename(columns={'Percent reduction in number of positive tests': 'Percent reduction in number of positive tests (CI method 1)'}) 
    #df = df.merge(df2[['FIT threshold (ug/g)', 'Model', 'Percent reduction in number of positive tests (CI method 1)', 'period']], how='left')
    #assert df.shape[0] == df2.shape[0]
    #df = df.rename(columns={'Percent reduction in number of positive tests': 'Percent reduction in number of positive tests (CI method 2)'}) 

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)

    first_cols = ['FIT threshold (ug/g)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                  'Percent reduction in number of positive tests',
                  'Positive tests per 1000 tests (model)', 
                  'Positive tests per 1000 tests (FIT)', 
                  'Delta sensitivity (model minus FIT)',
                  'Sensitivity model (%)',
                  'Sensitivity FIT (%)',
                  'Model threshold (%)']

    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]

    dfnew = dfnew.drop(labels = ['period'], axis=1)
    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())


    ## And CI method 1 separately
    fname = 'reformat_sens_fit_gain.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]

    # Dropping rows where the FIT test was not applied at original threshold, but at same sensitivity as original FIT
    df = df.loc[df.Model != 'FIT test']  
    df = df.merge(df_num, how='left', on='period')

    # Dropping rows where the FIT test was not applied at original threshold, but at same sensitivity as original FIT
    df = df.loc[df.Model != 'FIT test']  

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)

    first_cols = ['FIT threshold (ug/g)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                  'Percent reduction in number of positive tests',
                  'Positive tests per 1000 tests (model)', 
                  'Positive tests per 1000 tests (FIT)', 
                  'Delta sensitivity (model minus FIT)',
                  'Sensitivity model (%)',
                  'Sensitivity FIT (%)',
                  'Model threshold (%)']

    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]

    dfnew = dfnew.drop(labels = ['period'], axis=1)
    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())
    #endregion

    # .... 3. Metrics at predicted risk ....
    # It can be misleading when models are not fully calibrated
    # e.g. if model outperforms FIT, it won't necessarily when FIT threshold is just decreased
    #region

    ## ........ Recal models
    fname = 'reformat_risk.csv'

    thr_risk = np.array([0.5, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models_recal)]
    df = df.loc[df.period.isin(periods)]
    df = df.loc[df['Predicted risk (%)'].isin(thr_risk)]
    df = df.loc[df['Predicted risk (%)'] <= 10]

    df = df.merge(df_num, how='left')

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)

    dfnew = dfnew.drop(labels = ['period'], axis=1)

    first_cols = ['Predicted risk (%)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                    'Positive tests per 1000 tests', 'Negative tests per 1000 tests', 
                    'Detected cancers per 1000 tests', 'Missed cancers per 1000 tests',
                    'False positive tests per 1000 tests', 'True negative tests per 1000 tests']
    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]

    fname2 = pre + '-recal_' + fname

    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2)


    ## ........ Nottingham cox at 0.6% thr
    # Include FIT threshold 10 for comparison
    fname = 'reformat_risk.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]
    df = df.loc[df['Predicted risk (%)'] == 0.6]

    c_drop = ['FIT threshold (ug/g)', 'Model threshold (approx)']
    for c in c_drop:
        if c in df_fit10.columns:
            df_fit10 = df_fit10.drop(labels=c, axis=1)

    df_fit10['Model'] = "FIT test ≥ 10"
    df = pd.concat(objs = [df_fit10, df], axis=0)
    df = df.sort_values(by=['period', 'Model'])
    df = df.reset_index(drop=True)

    df = df.drop(labels=['Num patients', 'Num cancers', 'Prevalence (%)'], axis=1)
    df = df.merge(df_num, how='left', on='period')

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)

    first_cols = ['Predicted risk (%)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                    'Positive tests per 1000 tests', 'Negative tests per 1000 tests', 
                    'Detected cancers per 1000 tests', 'Missed cancers per 1000 tests',
                    'False positive tests per 1000 tests', 'True negative tests per 1000 tests']
    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]

    dfnew.loc[dfnew.Model == "FIT test ≥ 10", 'Predicted risk (%)'] = "-"

    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())

    #endregion

    # .... 4. Metrics at sens ...
    #region
    fname = 'reformat_sens.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]
    df['Sensitivity (%)'] = df['Sensitivity (%)'].round(2)

    df = df.merge(df_num, how='left', on='period')

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)

    first_cols = ['Sensitivity (%)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model', 
                  'Positive tests per 1000 tests', 'Negative tests per 1000 tests', 
                  'Detected cancers per 1000 tests', 'Missed cancers per 1000 tests',
                  'False positive tests per 1000 tests', 'True negative tests per 1000 tests',
                  'Threshold approx (%)']
    dfnew = dfnew[first_cols + [c for c in dfnew.columns if c not in first_cols]]
    
    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())

    #endregion

    # .... 5. Global disc ...
    #region

    fname = 'reformat_discrimination.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]
    df = df.merge(df_num, how='left', on='period')

    first_cols = ['Model', 'Num patients', 'Num cancers', 'Prevalence (%)']
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)

    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())
    
    #endregion

    # .... 6. Global cal ...
    #region

    fname = 'reformat_calibration_metrics.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models + models_recal)]
    df = df.loc[df.period.isin(periods)]
    df = df.merge(df_num, how='left', on='period')

    first_cols = ['Model', 'Num patients', 'Num cancers', 'Prevalence (%)']
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)

    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())
    

    #endregion

    # .... 7. DCA ...
    #region

    fname = 'reformat_dca.csv'

    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models + models_recal + ['Test all', 'Test none'])]
    df = df.loc[df.period.isin(periods)]
    df = df.merge(df_num.drop(labels=['Prevalence (%)'], axis=1), how='left', on='period')

    first_cols = ['Predicted risk (%)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model',
                  'Net benefit', 'Net intervention avoided', 
                  'Positive tests per 1000 tests', 
                  'Detected cancers per 1000 tests', 'Missed cancers per 1000 tests',
                  'False positive tests per 1000 tests'
                  ]
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
        
    dfnew = dfnew.drop(labels = ['period', 'Model group'], axis=1)

    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())
    #endregion

    
    # .... 8. Test reduction based on sens ...
    fname = 'reformat_sens_gain.csv'
    
    df = pd.read_csv(data_path / fname)
    df = df.loc[df.Model.isin(models)]
    df = df.loc[df.period.isin(periods)]
    df = df.merge(df_num, how='left', on='period')

    first_cols = ['Sensitivity (%)', 'Num patients', 'Num cancers', 'Prevalence (%)', 'Model'
                  ]
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    dfnew = pd.DataFrame()
    for p in periods:
        if p in df.period.unique():
            dfsub = df.loc[df.period == p]
            row = [time_label[p]] + [''] * (dfsub.shape[1] - 1)
            row = pd.DataFrame([row], columns=dfsub.columns)
            dfnew = pd.concat(objs=[dfnew, row, dfsub], axis=0)
    
    fname2 = pre + fname
    dfnew.to_csv(data_path / fname2, index=False)
    print("\nSaved", fname2, "\n...with columns", dfnew.columns.tolist())


# Run path
run_path = PROJECT_ROOT / 'results' / 'agg'
dirs = os.listdir(run_path)

# Do it with two sets of models
model_set = [['Nottingham-Cox', 'FIT test'], 
             ['Nottingham-lr', 'Nottingham-Cox', 'Nottingham-lr-boot', 'Nottingham-Cox-boot', 'FIT test']]
recal_model_set = [['Nottingham-Cox-quant', 'FIT-spline'], 
                   ['Nottingham-lr-quant', 'Nottingham-Cox-quant', 'FIT-spline']]
pre_set = ['model-cox_', 
           'model-all_']


# ---- Time cut analysis ---- 
dirs = os.listdir(run_path)
dirs = [d for d in dirs if d.startswith('timecut')]

periods = ['precovid', 'covid', 'post1', 'post2', 'post3', 'post4']

time_label = {'precovid': 'Pre-COVID (2017/01 - 2020/02)',
              'covid': 'COVID (2020/03 - 2021/04)' ,
              'post1': 'Post-COVID (2021/05 - 2021/12)',
              'post2': '2022 H1 (2022/01 - 2022/06)',
              'post3': '2022 H2 (2022/07 - 2022/12)',
              'post4': '2023 H1 (2023/01 - 2023/06)',
              'post3-buffcomm': '2022 H2 buffer only (2022/07 - 2022/12)',
              'post4-buffcomm': '2023 H1 buffer only (2023/01 - 2023/06)'
              }

# Loop over directories (fu 180 and fu 365)
for d in dirs:
    print(d)
    data_path = run_path / d

    # Loop over models 
    for pre, models, models_recal in zip(pre_set, model_set, recal_model_set):
        second_reformat_disc_cal(data_path, pre, models, periods, time_label)


# ---- Prepost analysis ---- 
dirs = os.listdir(run_path)
dirs = [d for d in dirs if d.startswith('prepost')]

periods = ['all', 'prebuff', 'bufftime', 'buffcomm']

time_label_365 = {
    'all': 'All data (2017/01 - 2023/02)',
    'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
    'bufftime': 'Buffer data from time (2021/10 - 2023/02)',
    'buffcomm': 'Buffer data from comment (2022/04 - 2023/02)'
    }

time_label_180 = {
    'all': 'All data (2017/01 - 2023/08)',
    'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
    'bufftime': 'Buffer data from time (2021/10 - 2023/08)',
    'buffcomm': 'Buffer data from comment (2022/04 - 2023/08)'
    }

# Loop over directories (fu 180 and fu 365)
for d in dirs:
    print(d)
    data_path = run_path / d

    if '365' in d:
        time_label = time_label_365
    else:
        time_label = time_label_180

    # Loop over models 
    for pre, models, models_recal in zip(pre_set, model_set, recal_model_set):
        second_reformat_disc_cal(data_path, pre, models, periods, time_label)
