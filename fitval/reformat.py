"""Reformat output tables"""
from fitval.models import model_labels
from fitval.boot import DISC_CI, CAL_CI, RISK_CI, SENS_CI, SENS_FIT_CI, SENS_FIT_GAIN_CI, DC_CI, SENS_FIT_2_CI, SENS_FIT_GAIN_2_CI, PR_GAIN_CI
# from fitval.strata import DISC_STRATA, CAL_STRATA
import pandas as pd
import numpy as np
import re
from pathlib import Path


# Reformatted discrimination, calibration, net benefit tables
DISC_PAPER = 'reformat_discrimination.csv'
CAL_PAPER = 'reformat_calibration_metrics.csv'
RISK_PAPER = 'reformat_risk.csv'
RISK1000 = 'reformat_risk1000.csv'
SENS_PAPER = 'reformat_sens.csv'
SENS_GAIN_PAPER = 'reformat_sens_gain.csv'
SENS_FIT_PAPER = 'reformat_sens_fit.csv'
SENS_FIT_GAIN_PAPER = 'reformat_sens_fit_gain.csv'
SENS_FIT_PAPER2 = 'reformat_sens_fit2.csv'
SENS_FIT_GAIN_PAPER2 = 'reformat_sens_fit_gain2.csv'
DC_PAPER = 'reformat_dca.csv'

# Reformatted tables for metrics in strata
DISC_STRATA_PAPER = 'reformat_discrimination_strata.csv'
CAL_STRATA_PAPER = 'reformat_calibration_metrics_strata.csv'

# Model groups
model_group = {'fit-lowess': 'fit', 
               'fit10': 'fit',
               'fit-spline': 'fit',
               'fit-ebm': 'fit',
               'fit-age': 'fit-age', 
               'fit-age-sex': 'fast',
               
               'nottingham-fit': 'nott',
               'nottingham-fit-age': 'nott',
               'nottingham-fit-age-sex': 'nott',

               'nottingham-lr': 'nott', 
               'nottingham-lr-boot': 'nott',
               'nottingham-cox': 'nott',
               'nottingham-cox-boot': 'nott',

               'nottingham-lr-quant': 'nott-recal', 
               'nottingham-lr-3.5': 'nott-recal',
               'nottingham-lr-platt': 'nott-recal',
               'nottingham-lr-iso': 'nott-recal',

               'nottingham-cox-quant': 'nott-recal', 
               'nottingham-cox-3.5': 'nott-recal',
               'nottingham-cox-platt': 'nott-recal',
               'nottingham-cox-iso': 'nott-recal',

               'nottingham-fit-platt': 'nott-recal',
               'nottingham-fit-quant': 'nott-recal',
               'nottingham-fit-3.5': 'nott-recal',

               'nottingham-fit-age-sex-platt': 'nott-recal',
               'nottingham-fit-age-sex-quant': 'nott-recal',
               'nottingham-fit-age-sex-3.5': 'nott-recal',

               'nottingham-fit-age-platt': 'nott-recal',
               'nottingham-fit-age-quant': 'nott-recal',
               'nottingham-fit-age-3.5': 'nott-recal',

               'all': 'net_all',
               'none': 'net_none'
               }


# ~ Helpers ~
def _reformat_ci(df, digits=3):
    df = df.drop(labels=['q025', 'q975'], axis=1)
    df[['metric_value', 'ci_low', 'ci_high']] = df[['metric_value', 'ci_low', 'ci_high']].round(digits)
    df['metric_value'] = df.metric_value.astype(str) + ' (' + df.ci_low.astype(str) + ', ' + df.ci_high.astype(str) + ')'
    df = df.drop(labels=['ci_low', 'ci_high'], axis=1)
    return df


def check_nan(df):
    for c in df.columns:
        test = df[c].isna()
        if test.any():
            print('\n----Column', c, 'contains nan')
            print(df.loc[test].iloc[0:3])


# ~ Main functions ~
def reformat_disc_cal(data_path: Path, save_path: Path, model_order: list = None):
    print('Reformatting discrimination and calibration tables...')

    if model_order is None:
        model_order = [
                       'fit', 'fit10', 'fit-spline', 'fit-ebm', 'fit-lowess', 'fit-age', 'fit-age-sex',
                       
                       # Nottingham FIT, age, sex moels
                       'nottingham-fit', 'nottingham-fit-age', 'nottingham-fit-age-sex',

                       # Nottingham 
                       'nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot',
                      
                        # Recalibrated models
                        'nottingham-fit-3.5', 'nottingham-fit-platt', 'nottingham-fit-quant',
                        'nottingham-fit-age-platt', 'nottingham-fit-age-quant', 'nottingham-fit-age-3.5',
                        'nottingham-fit-age-sex-platt', 'nottingham-fit-age-sex-quant', 'nottingham-fit-age-sex-3.5',
                        'nottingham-lr-quant', 'nottingham-lr-3.5', 'nottingham-lr-platt', 'nottingham-lr-iso',
                        'nottingham-cox-quant', 'nottingham-cox-3.5', 'nottingham-cox-platt', 'nottingham-cox-iso',

                        'all', 'none'
                       ]
    
    # ---- 1. Global discrimination metrics ----
    #region
    df = pd.read_csv(data_path / DISC_CI)
    check_nan(df)
    assert df.shape == df.drop_duplicates().shape

    # Reformat CI
    df[['metric_value', 'ci_low', 'ci_high']] *= 100  # Rescale to %
    df = _reformat_ci(df, digits=2)

    # To wide format
    df = df.pivot(index='model_name', columns='metric_name', values='metric_value').reset_index()

    # Reorder models
    df = pd.concat(objs=[df.loc[df.model_name == m] for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)

    # Add 'section breaks'
    df0 = df.loc[df.model_group == 'fit']
    df1 = df.loc[df.model_group == 'nott']
    label1 = pd.DataFrame([['Original models'] + [''] * (df.shape[1] - 1)], columns=df0.columns)
    df2 = df.loc[df.model_group == 'nott-recal']
    label2 = pd.DataFrame([['Recalibrated models'] + [''] * (df.shape[1] - 1)], columns=df0.columns)
    df = pd.concat(objs=[df0, label1, df1, label2, df2], axis=0)
    
    # Tidy model and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'model_name': 'Model', 'ap': 'Average precision (%)', 'auroc': 'c-statistic (%)',
                            'model_group': 'Model group'})

    df.to_csv(save_path / DISC_PAPER, index=False)
    #endregion

    # ---- 2. Global calibration metrics ----
    #region
    df = pd.read_csv(data_path / CAL_CI)
    check_nan(df)
    assert df.shape == df.drop_duplicates().shape

    df = _reformat_ci(df, digits=2)
    df = df.pivot(index='model_name', columns='metric_name', values='metric_value').reset_index()
    df = df[['model_name', 'event_rate', 'mean_risk', 'oe_ratio', 'log_intercept', 'log_slope']]
    df = pd.concat(objs=[df.loc[df.model_name == m] for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)

    ## Add 'section breaks'
    df0 = df.loc[df.model_group == 'fit']
    df1 = df.loc[df.model_group == 'nott']
    label1 = pd.DataFrame([['Original models'] + [''] * (df.shape[1] - 1)], columns=df0.columns)
    df2 = df.loc[df.model_group == 'nott-recal']
    label2 = pd.DataFrame([['Recalibrated models'] + [''] * (df.shape[1] - 1)], columns=df0.columns)
    df = pd.concat(objs=[df0, label1, df1, label2, df2], axis=0)

    ## Tidy model and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'model_name': 'Model', 'event_rate': 'Event rate (%)', 'mean_risk': 'Mean risk (%)',
                            'oe_ratio': 'O/E ratio', 'log_intercept': 'Log intercept', 'log_slope': 'Log slope',
                            'model_group': 'Model group'}) 

    df.to_csv(save_path / CAL_PAPER, index=False)
    #endregion

    # ---- 3. Metrics at predefined risk thresholds ----
    #region

    ## Read data and rescale to percentage
    df = pd.read_csv(data_path / RISK_CI)
    check_nan(df)
    assert df.shape[0] == df.drop_duplicates().shape[0]

    mask = df.metric_name.isin(['sens', 'spec', 'ppv', 'npv'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100

    mask = df.model_name != 'fit'
    df.loc[mask, 'thr'] = (df.loc[mask, 'thr'] * 100).round(4) # If round to less than 3 digits, non-unique thr values
    
    ## Get tp, tn, fp, fn counts per 1000 tests additionally
    npat = df.loc[df.metric_name.isin(['pp', 'pn'])].pivot(index=['thr', 'model_name'], columns=['metric_name'], values=['metric_value'])
    npat = npat.sum(axis=1).iloc[0]  # Number of patients (Tests) 
    for metric_name in ['pp', 'pn', 'tp', 'tn', 'fp', 'fn']:
        dfsub = df.loc[df.metric_name == metric_name].copy()
        dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1000 / npat)
        dfsub.metric_name = metric_name + '1000'
        df = pd.concat(objs=[df, dfsub], axis=0)

    ## To wider format with CI
    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['thr', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df[['thr', 'model_name', 'sens', 'spec', 'ppv', 'npv', 'pp', 'pn', 'tp', 'fn', 'fp', 'tn', 'pp_per_cancer',
             'pp1000', 'pn1000', 'tp1000', 'fn1000', 'fp1000', 'tn1000']]

    ## Reorder rows
    thr = df.thr.unique()
    df = pd.concat(objs=[df.loc[(df.thr == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)
    
    # Tidy model name and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'thr': 'Predicted risk (%)',
                            'model_name': 'Model',
                            'sens': 'Sensitivity (%)', 
                            'spec': 'Specificity (%)', 
                            'npv':'NPV (%)', 
                            'ppv': 'PPV (%)',
                            'pp': 'Positive tests', 
                            'pp_per_cancer': 'Positive tests per cancer', 
                            'pn': 'Negative tests',
                            'tp': 'Detected cancers', 
                            'fp': 'False positive tests', 
                            'tn': 'True negative tests', 
                            'fn': 'Missed cancers',
                            'pp1000': 'Positive tests per 1000 tests', 
                            'pn1000': 'Negative tests per 1000 tests',
                            'tp1000': 'Detected cancers per 1000 tests', 
                            'fp1000': 'False positive tests per 1000 tests', 
                            'tn1000': 'True negative tests per 1000 tests', 
                            'fn1000': 'Missed cancers per 1000 tests', 
                            'model_group': 'Model group'})

    df.to_csv(save_path / RISK_PAPER, index=False)

    #endregion

    # ---- 4. Metrics at sensitivity ----
    # Could get more from pr_int data, but perhaps not necessary 
    # Note: at lower sensiivities, like 50%, the metrics of FIT can be nan
    #       this happens when e.g. at max threshold the sensitivity is greater than 50%.
    #region

    ## Read data
    df = pd.read_csv(data_path / SENS_CI)
    check_nan(df)

    ## Rescale to %
    mask = df.metric_name.isin(['spec', 'ppv', 'npv'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   
    df.loc[(df.model_name != 'fit') & (df.metric_name == 'thr'), ['metric_value', 'ci_low', 'ci_high']] *= 100
    df.sens = (df.sens * 100).round(2)

    ## Get tp, tn, fp, fn counts per 1000 tests additionally
    npat = df.loc[df.metric_name.isin(['pp', 'pn'])].pivot(index=['sens', 'model_name'], columns=['metric_name'], values=['metric_value'])
    npat = npat.sum(axis=1).iloc[0]  # Number of patients (Tests) 
    for metric_name in ['pp', 'pn', 'tp', 'tn', 'fp', 'fn']:
        dfsub = df.loc[df.metric_name == metric_name].copy()
        dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1000 / npat)
        dfsub.metric_name = metric_name + '1000'
        df = pd.concat(objs=[df, dfsub], axis=0)

    ## To wider format with CI
    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['sens', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df[['sens', 'model_name', 'spec', 'ppv', 'npv', 'pp', 'pn', 'tp', 'fn', 'fp', 'tn', 'pp_per_cancer',
             'pp1000', 'pn1000', 'tp1000', 'fn1000', 'fp1000', 'tn1000', 'thr']]

    ## Reorder rows
    thr = df.thr.unique()
    df = pd.concat(objs=[df.loc[(df.thr == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)

    ## Tidy model name and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'thr': 'Threshold approx (%)',
                            'model_name': 'Model',
                            'sens': 'Sensitivity (%)', 
                            'spec': 'Specificity (%)', 
                            'npv':'NPV (%)', 
                            'ppv': 'PPV (%)',
                            'pp': 'Positive tests', 
                            'pp_per_cancer': 'Positive tests per cancer', 
                            'pn': 'Negative tests',
                            'tp': 'Detected cancers', 
                            'fp': 'False positive tests', 
                            'tn': 'True negative tests', 
                            'fn': 'Missed cancers',
                            'pp1000': 'Positive tests per 1000 tests', 
                            'pn1000': 'Negative tests per 1000 tests',
                            'tp1000': 'Detected cancers per 1000 tests', 
                            'fp1000': 'False positive tests per 1000 tests', 
                            'tn1000': 'True negative tests per 1000 tests', 
                            'fn1000': 'Missed cancers per 1000 tests', 
                            'model_group': 'Model group'})

    df.to_csv(save_path / SENS_PAPER, index=False)
    #endregion

    # ---- 5. Metrics at sensitivity corresponding to FIT thr ----
    #region
    for in_file, out_file in zip([SENS_FIT_CI, SENS_FIT_2_CI], 
                                 [SENS_FIT_PAPER, SENS_FIT_PAPER2]):
        print(in_file, out_file)
        
        ## Read data
        df = pd.read_csv(data_path / in_file)
        check_nan(df)  ## It's OK for max_sens to be nan here, it is not relevant anyway, as not computed for FIT at thr 2/10 etc

        ## Fix incorrectly assigned model_name if present
        df.loc[df.model == 'fit', 'model_name'] = 'fit'
        df = df.loc[(df.model == 'model') & (df.model_name != 'fit')]  # rm FIT from models here

        ## Rescale to %
        mask = df.metric_name.isin(['sens', 'spec', 'ppv', 'npv'])
        df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   
        df.loc[(df.model_name != 'fit') & (df.metric_name == 'thr'), ['metric_value', 'ci_low', 'ci_high']] *= 100

        ## Get tp, tn, fp, fn counts per 1000 tests additionally
        npat = df.loc[df.metric_name.isin(['pp', 'pn'])].pivot(index=['thr_fit', 'model', 'model_name'], columns=['metric_name'], values=['metric_value'])
        npat = npat.sum(axis=1).iloc[0]  # Number of patients (Tests) 
        for metric_name in ['pp', 'pn', 'tp', 'tn', 'fp', 'fn']:
            dfsub = df.loc[df.metric_name == metric_name].copy()
            dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1000 / npat)
            dfsub.metric_name = metric_name + '1000'
            df = pd.concat(objs=[df, dfsub], axis=0)

        ## Drop rows with nan
        df.isna().sum()
        df.loc[df.metric_value.isna(), 'metric_name'].unique()
        df = df.loc[~df.metric_value.isna()]

        ## Drop duplicates just in case
        df0 = df.loc[df.model == 'fit'].drop_duplicates(subset=['thr_fit', 'model', 'metric_name'])
        df1 = df.loc[df.model != 'fit']
        df1 = df1.loc[df1.model_name != 'fit']  # Drop interpolated FIT test (as not the point of this table)
        df = pd.concat(objs=[df0, df1], axis=0)

        ## To wide format
        df = _reformat_ci(df, digits=2)
        df = df.pivot(index=['thr_fit', 'model', 'model_name'], columns='metric_name', values='metric_value').reset_index()
        df = df[['thr_fit', 'model_name', 'sens', 'spec', 'ppv', 'npv', 'pp', 'pn', 'tp', 'fn', 'fp', 'tn', 'pp_per_cancer',
                'pp1000', 'pn1000', 'tp1000', 'fn1000', 'fp1000', 'tn1000', 'thr']]

        ## Reorder
        thr = df.sens.unique()
        df = pd.concat(objs=[df.loc[(df.sens == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
        df['Model group'] = df.model_name.replace(model_labels)

        ## Tidy model name and column names
        df.model_name = df.model_name.replace(model_labels)
        df = df.rename(columns={'thr_fit': 'FIT threshold (ug/g)',
                                'thr': 'Threshold approx (%)',
                                'model_name': 'Model',
                                'sens': 'Sensitivity (%)', 
                                'spec': 'Specificity (%)', 
                                'npv':'NPV (%)', 
                                'ppv': 'PPV (%)',
                                'pp': 'Positive tests', 
                                'pp_per_cancer': 'Positive tests per cancer', 
                                'pn': 'Negative tests',
                                'tp': 'Detected cancers', 
                                'fp': 'False positive tests', 
                                'tn': 'True negative tests', 
                                'fn': 'Missed cancers',
                                'pp1000': 'Positive tests per 1000 tests', 
                                'pn1000': 'Negative tests per 1000 tests',
                                'tp1000': 'Detected cancers per 1000 tests', 
                                'fp1000': 'False positive tests per 1000 tests', 
                                'tn1000': 'True negative tests per 1000 tests', 
                                'fn1000': 'Missed cancers per 1000 tests', 
                                'model_group': 'Model group'})
        df.to_csv(save_path / out_file, index=False)
    #endregion

    # ---- 6. Metrics at sensitivity corresponding to FIT thr: gain ----
    # Note: unintentionally ran this with model = 'fit'
    # This means a threshold was learned for fit that yields same sens which is slightly dif from actual fit thr
    # Also note that model thr not given here, BUT can be taken from the previous table.
    #region
    for thr_file, in_file, out_file in zip([SENS_FIT_CI, SENS_FIT_2_CI], 
                                           [SENS_FIT_GAIN_CI, SENS_FIT_GAIN_2_CI], 
                                           [SENS_FIT_GAIN_PAPER, SENS_FIT_GAIN_PAPER2]):
        print(thr_file, in_file, out_file)

        ## Get model thresholds corresponding to each level of sensitivity
        df0 = pd.read_csv(data_path / thr_file)
        df0 = df0.loc[(df0.metric_name == 'thr') & (df0.model == 'model')]
        df0 = df0.loc[df0.model_name != 'fit']  # rm FIT just in case if it is included

        ## Read data
        df = pd.read_csv(data_path / in_file)
        df = pd.concat(objs=[df0, df], axis=0)
        df = df.loc[(df.model == 'model') & (df.model_name != 'fit')]  # rm FIT just in case if it is included

        ## Rescale to %
        mask = df.metric_name.isin(['precision_gain', 'proportion_reduction_tests', 'delta_sens',
                                    'ppv_mod', 'ppv_fit', 'sens_mod', 'sens_fit'])
        df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   
        df.loc[(df.model_name != 'fit') & (df.metric_name == 'thr'), ['metric_value', 'ci_low', 'ci_high']] *= 100

        ## Get pp per 1000 tests additionally (npat computeds before)
        for metric_name, new_name in zip(['pp_mod', 'pp_fit'], ['pp_mod_1000', 'pp_fit_1000']):
            dfsub = df.loc[df.metric_name == metric_name].copy()
            dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1/npat * 1000)
            dfsub.metric_name = new_name
            df = pd.concat(objs=[df, dfsub], axis=0)

        ## To wide format
        df = _reformat_ci(df, digits=2)
        df = df.drop_duplicates()
        df = df.pivot(index=['thr_fit', 'model', 'model_name'], columns='metric_name', values='metric_value').reset_index()
        df = df[['thr_fit', 'model_name', 'proportion_reduction_tests', 'delta_sens', 'pp_mod', 'pp_fit', 
                 'sens_mod', 'sens_fit', 'pp_mod_1000', 'pp_fit_1000', 'precision_gain', 'ppv_mod', 'ppv_fit',
                 'thr']]

        ## Reorder
        thr = df.thr_fit.unique()
        df = pd.concat(objs=[df.loc[(df.thr_fit == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)

        ## Tidy model name and column names
        df.model_name = df.model_name.replace(model_labels)
        df = df.rename(columns={'thr_fit': 'FIT threshold (ug/g)', 
                                'thr': 'Model threshold (%)', 
                                'model_name': 'Model', 
                                'precision_gain': 'Gain in PPV (% scale)', 
                                'proportion_reduction_tests': 'Percent reduction in number of positive tests',
                                'delta_sens': 'Delta sensitivity (model minus FIT)',
                                'ppv_mod': 'PPV model (%)', 
                                'ppv_fit': 'PPV FIT (%)',
                                'pp_mod': 'Positive tests (model)', 
                                'pp_fit': 'Positive tests (FIT)',
                                'pp_mod_1000': 'Positive tests per 1000 tests (model)', 
                                'pp_fit_1000': 'Positive tests per 1000 tests (FIT)',
                                'sens_mod': 'Sensitivity model (%)', 
                                'sens_fit': 'Sensitivity FIT (%)'})
        df.to_csv(save_path / out_file, index=False)
    #endregion

    # ---- 7. Net benefit ----
    #region

    ## Read data and rescale to percentage
    df = pd.read_csv(data_path / DC_CI)
    check_nan(df)
    assert df.shape[0] == df.drop_duplicates().shape[0]

    ## Rescale
    mask = df.metric_name.isin(['prevalence', 'test_pos_rate', 'tp_rate', 'fp_rate', 'fn_rate'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100
    df = df.rename(columns={'threshold': 'thr'})
    df.thr *= 100
    df.thr = df.thr.round(5)  # If round to less than 3 digits, non-unique thr values

    ## Get tp, tn, fp, fn counts
    npat = df.loc[df.metric_name == 'n', 'metric_value'].iloc[0]
    for metric_name, new_name in zip(['test_pos_rate', 'tp_rate', 'fp_rate', 'fn_rate'],
                                     ['pp', 'tp', 'fp', 'fn']):
        dfsub = df.loc[df.metric_name == metric_name].copy()
        dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1/100 * npat)
        dfsub.metric_name = new_name
        df = pd.concat(objs=[df, dfsub], axis=0)

    ## Get tp, tn, fp, fn counts per 1000 tests additionally
    npat = df.loc[df.metric_name == 'n', 'metric_value'].iloc[0]
    for metric_name in ['pp', 'tp', 'fp', 'fn']:
        dfsub = df.loc[df.metric_name == metric_name].copy()
        dfsub[['metric_value', 'ci_low', 'ci_high']] *= (1/npat * 1000)
        dfsub.metric_name = metric_name + '1000'
        df = pd.concat(objs=[df, dfsub], axis=0)

    ## To wider format with CI
    df.shape
    df = df.drop_duplicates(subset=['model_name', 'thr', 'metric_name'])
    df.shape

    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['thr', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df[['thr', 'model_name', 'n', 'prevalence', 'pp', 'tp', 'fn', 'fp',
             'net_benefit', 'net_intervention_avoided',
             'pp1000', 'tp1000', 'fp1000', 'fn1000']]

    ## Reorder rows
    thr = df.thr.unique()
    df = pd.concat(objs=[df.loc[(df.thr == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)

    # Tidy model name and column names
    df.model_name = df.model_name.replace(model_labels)
    df.model_name = df.model_name.replace({'all': 'Test all', 'none': 'Test none', 'fit10': 'FIT ≥ 10'})
    df = df.rename(columns={'thr': 'Predicted risk (%)',
                            'model_name': 'Model',
                            'n': 'Number of tests',
                            'prevalence': 'Prevalence (%)',
                            'test_pos_rate': 'Positive tests (%)',
                            'tp_rate': 'Sensitivity (%)',
                            'fp_rate': 'False positive tests (%)',
                            'fn_rate': 'False negative tests (%)',
                            'net_benefit': 'Net benefit',
                            'net_intervention_avoided': 'Net intervention avoided',
                            'harm': 'harm',
                            'tp_rate1000': 'Sensitivity (%)',
                            'fp_rate1000': 'False positive tests (%)',
                            'fn_rate1000': 'False negative tests (%)',
                            'sens': 'Sensitivity (%)', 
                            'spec': 'Specificity (%)', 
                            'npv':'NPV (%)', 
                            'ppv': 'PPV (%)',
                            'pp': 'Positive tests', 
                            'pp_per_cancer': 'Positive tests per cancer', 
                            'pn': 'Negative tests',
                            'tp': 'Detected cancers', 
                            'fp': 'False positive tests', 
                            'tn': 'True negative tests', 
                            'fn': 'Missed cancers',
                            'pp1000': 'Positive tests per 1000 tests', 
                            'pn1000': 'Negative tests per 1000 tests',
                            'tp1000': 'Detected cancers per 1000 tests', 
                            'fp1000': 'False positive tests per 1000 tests', 
                            'tn1000': 'True negative tests per 1000 tests', 
                            'fn1000': 'Missed cancers per 1000 tests', 
                            'model_group': 'Model group'})

    df.to_csv(save_path / DC_PAPER, index=False)
    #endregion

    # ---- 8. Test reduction at sens ----
    #regions

    ## Read data
    df = pd.read_csv(data_path / PR_GAIN_CI)
    check_nan(df)

    df0 = pd.read_csv(data_path / SENS_CI)
    df0.sens = (df0.sens * 100).round(2)

    ## Rescale to %
    mask = df.metric_name.isin(['precision', 'precision_fit', 'precision_gain', 'proportion_reduction_tests'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   
    df.recall = (df.recall * 100).round(2)
    df = df.rename(columns={'recall': 'sens'})

    ## To wider format with CI
    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['sens', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df.rename(columns={'precision': 'ppv_mod', 'recall': 'sens', 'precision_fit': 'ppv_fit'})
    df = df[['sens', 'model_name', 'proportion_reduction_tests', 'ppv_mod', 'ppv_fit', 'precision_gain']]

    df = df.loc[df.sens.isin(df0.sens)]

    ## Reorder rows
    thr = df.sens.unique()
    df = pd.concat(objs=[df.loc[(df.sens == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
    df['model_group'] = df.model_name.replace(model_group)

    ## Tidy model name and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'thr': 'Threshold approx (%)',
                            'model_name': 'Model',
                            'ppv_mod': 'PPV model (%)',
                            'ppv_fit': 'PPV FIT (%)',
                            'precision_gain': 'Gain in PPV (% scale)',
                            'proportion_reduction_tests': 'Percent reduction in number of positive tests',
                            'sens': 'Sensitivity (%)', 
                            'spec': 'Specificity (%)', 
                            'npv':'NPV (%)', 
                            'ppv': 'PPV (%)',
                            'pp': 'Positive tests', 
                            'pp_per_cancer': 'Positive tests per cancer', 
                            'pn': 'Negative tests',
                            'tp': 'Detected cancers', 
                            'fp': 'False positive tests', 
                            'tn': 'True negative tests', 
                            'fn': 'Missed cancers',
                            'pp1000': 'Positive tests per 1000 tests', 
                            'pn1000': 'Negative tests per 1000 tests',
                            'tp1000': 'Detected cancers per 1000 tests', 
                            'fp1000': 'False positive tests per 1000 tests', 
                            'tn1000': 'True negative tests per 1000 tests', 
                            'fn1000': 'Missed cancers per 1000 tests', 
                            'model_group': 'Model group'})

    
    df.to_csv(save_path / SENS_GAIN_PAPER, index=False)
    #endregion

    print('Reformat complete.')
    

def reformat_strata(run_path):
    raise NotImplementedError
    
    df = pd.read_csv(run_path / DISC_STRATA)
    df[['metric_value', 'ci_low', 'ci_high']] *= 100
    df = _reformat_strata(df, digits=1)
    df.to_csv(run_path / DISC_STRATA_PAPER, index=False)

    df = pd.read_csv(run_path / CAL_STRATA)
    df = _reformat_strata(df)
    df = df[['strata_variable', 'strata_value', 'strata_n', 'strata_n_crc', 'model_name',
             'event_rate', 'mean_risk', 'oe_ratio', 'log_intercept', 'log_slope', 'model_group']]
    df.to_csv(run_path / CAL_STRATA_PAPER, index=False)


def _reformat_strata(df, digits=3):
    raise NotImplementedError

    model_order = ['fit', 'fit10', 'fit-spline', 'fit-lowess', 'fit-age', 'fit-age-sex',

                    'nottingham-fit', 'nottingham-fit-platt', 'nottingham-fit-quant', 'nottingham-fit-3.5',
                    'nottingham-fit-age', 'nottingham-fit-age-platt', 'nottingham-fit-age-quant', 'nottingham-fit-age-3.5',
                    'nottingham-fit-age-sex', 'nottingham-fit-age-sex-platt', 'nottingham-fit-age-sex-quant', 'nottingham-fit-age-sex-3.5',

                    'nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot',
                    'nottingham-lr-quant', 'nottingham-lr-3.5', 'nottingham-lr-platt', 'nottingham-lr-iso']

    df = _reformat_ci(df, digits=digits)
    df = df.pivot(index=['strata_variable', 'strata_value', 'strata_n', 'strata_n_crc', 'model_name'],
                  columns='metric_name', values='metric_value').reset_index()
    df['model_group'] = df.model_name.replace(model_group)

    tmp = pd.DataFrame()
    for var in df.strata_variable.unique():
        dfsub = df.loc[df.strata_variable == var]
        val = dfsub.strata_value.sort_values().unique()
        for v in val:
            dfsub2 = dfsub.loc[dfsub.strata_value == v]
            dfsub2 = pd.concat(objs=[dfsub2.loc[dfsub2.model_name == m] for m in model_order if m in dfsub2.model_name.unique()], axis=0)
            tmp = pd.concat(objs=[tmp, dfsub2], axis=0)
    return tmp
    