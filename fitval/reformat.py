"""Reformat output tables"""
from fitval.boot import DISC_CI, CAL_CI, RISK_CI, SENS_CI, SENS_FIT_CI, DC_CI, SENS_FIT_2_CI, PR_GAIN_CI
import pandas as pd
from pathlib import Path


# Reformatted discrimination, calibration, net benefit tables
DISC_PAPER = 'reformat_discrimination.csv'
CAL_PAPER = 'reformat_calibration.csv'
RISK_PAPER = 'reformat_metrics_at_risk.csv'
SENS_PAPER = 'reformat_metrics_at_sens.csv'
SENS_FIT_PAPER = 'reformat_metrics_at_sens_fit.csv'
SENS_FIT_PAPER2 = 'reformat_metrics_at_fit_and_mod_thresholds.csv'
DC_PAPER = 'reformat_dca.csv'
SENS_GAIN_PAPER = 'reformat_pr_gain.csv'


metric_names = {
    'ap': 'Average precision (%)', 
    'auroc': 'c-statistic (%)',
    'event_rate': 'Event rate (%)', 
    'mean_risk': 'Mean risk (%)',
    'oe_ratio': 'O/E ratio', 
    'log_intercept': 'Log intercept', 
    'log_slope': 'Log slope',
    'thr_fit': 'FIT threshold (ug/g)',
    'thr': 'Model threshold (%)',
    'thr_mod': 'Model threshold (%)',
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

    'sens_fit': 'Sensitivity (%), FIT', 
    'spec_fit': 'Specificity (%), FIT', 
    'npv_fit':'NPV (%), FIT', 
    'ppv_fit': 'PPV (%), FIT',
    'pp_fit': 'Positive tests (FIT)', 
    'pp_per_cancer_fit': 'Positive tests per cancer (FIT)', 
    'pn_fit': 'Negative tests (FIT)',
    'tp_fit': 'Detected cancers (FIT)', 
    'fp_fit': 'False positive tests (FIT)', 
    'tn_fit': 'True negative tests (FIT)', 
    'fn_fit': 'Missed cancers (FIT)',
    'pp1000_fit': 'Positive tests per 1000 tests (FIT)', 
    'pn1000_fit': 'Negative tests per 1000 tests (FIT)',
    'tp1000_fit': 'Detected cancers per 1000 tests (FIT)', 
    'fp1000_fit': 'False positive tests per 1000 tests (FIT)', 
    'tn1000_fit': 'True negative tests per 1000 tests (FIT)', 
    'fn1000_fit': 'Missed cancers per 1000 tests (FIT)',

    'sens_mod': 'Sensitivity (%), model', 
    'spec_mod': 'Specificity (%), model', 
    'npv_mod':'NPV (%), model', 
    'ppv_mod': 'PPV (%), model',
    'pp_mod': 'Positive tests (model)', 
    'pp_per_cancer_mod': 'Positive tests per cancer (model)', 
    'pn_mod': 'Negative tests (model)',
    'tp_mod': 'Detected cancers (model)', 
    'fp_mod': 'False positive tests (model)', 
    'tn_mod': 'True negative tests (model)', 
    'fn_mod': 'Missed cancers (model)',
    'pp1000_mod': 'Positive tests per 1000 tests (model)', 
    'pn1000_mod': 'Negative tests per 1000 tests (model)',
    'tp1000_mod': 'Detected cancers per 1000 tests (model)', 
    'fp1000_mod': 'False positive tests per 1000 tests (model)', 
    'tn1000_mod': 'True negative tests per 1000 tests (model)', 
    'fn1000_mod': 'Missed cancers per 1000 tests (model)',


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
def reformat_disc_cal(data_path: Path, save_path: Path, model_labels: dict = None):
    print('Reformatting discrimination and calibration tables...')

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
    models = df.model_name.unique().tolist()
    model_order = ['all', 'none', 'fit', 'fit10', 'fit-spline'] + [m for m in models if m not in ['fit', 'fit10', 'fit-spline', 'all', 'none']]
    df = pd.concat(objs=[df.loc[df.model_name == m] for m in model_order if m in df.model_name.unique()], axis=0)

    # Tidy model and column names
    if model_labels is not None:
        df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns=metric_names)
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

    # Tidy model and column names
    if model_labels is not None:
        df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns=metric_names) 
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
    
    ## To wider format with CI
    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['thr', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df[['thr', 'model_name', 'sens', 'spec', 'ppv', 'npv', 'pp', 'pn', 'tp', 'fn', 'fp', 'tn', 'pp_per_cancer',
             'pp1000', 'pn1000', 'tp1000', 'fn1000', 'fp1000', 'tn1000']]

    ## Reorder rows
    thr = df.thr.drop_duplicates().sort_values().unique()
    df = pd.concat(objs=[df.loc[(df.thr == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)
    
    # Tidy model name and column names
    if model_labels is not None:
        df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns=metric_names)
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

    ## To wider format with CI
    df = _reformat_ci(df, digits=2)
    df = df.pivot(index=['sens', 'model_name'], columns='metric_name', values='metric_value').reset_index()
    df = df[['sens', 'model_name', 'spec', 'ppv', 'npv', 'pp', 'pn', 'tp', 'fn', 'fp', 'tn', 'pp_per_cancer',
             'pp1000', 'pn1000', 'tp1000', 'fn1000', 'fp1000', 'tn1000', 'thr']]

    ## Reorder rows
    thr = df.sens.drop_duplicates().sort_values().unique()
    df = pd.concat(objs=[df.loc[(df.sens == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)

    ## Tidy model name and column names
    if model_labels is not None:
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

        ## Rescale to %
        mask = df.metric_name.isin(['sens', 'spec', 'ppv', 'npv',
                                    'sens_fit', 'spec_fit', 'ppv_fit', 'npv_fit',
                                    'sens_mod', 'spec_mod', 'ppv_mod', 'npv_mod'])
        df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   
        df.loc[(df.model_name != 'fit') & (df.metric_name.isin(['thr', 'thr_fit', 'thr_mod'])), ['metric_value', 'ci_low', 'ci_high']] *= 100
        if 'thr_mod' in df.columns:
            df.thr_mod = df.thr_mod * 100

        ## Drop rows with nan
        df.isna().sum()
        df.loc[df.metric_value.isna(), 'metric_name'].unique()
        df = df.loc[~df.metric_value.isna()]

        ## To wide format
        df = _reformat_ci(df, digits=2)
        if 'thr_mod' in df.columns:
            df = df.pivot(index=['thr_fit', 'thr_mod', 'model_name'], columns='metric_name', values='metric_value').reset_index()
        else:
            df = df.pivot(index=['thr_fit', 'model_name'], columns='metric_name', values='metric_value').reset_index()
        cols = ['thr_fit', 'thr_mod', 'model_name', 'pp_mod', 'pp_fit', 'test_red',
                'sens_mod', 'sens_fit', 'delta_sens', 'tp_mod', 'tp_fit', 'delta_tp',
                'ppv_mod', 'ppv_fit', 'delta_ppv', 'pp_per_cancer_mod', 'pp_per_cancer_fit',
                'pp1000_mod', 'pp1000_fit', 'tp1000_mod', 'tp1000_fit'
                ]
        cols_use = [c for c in cols if c in df.columns]
        df = df[cols_use]

        ## Reorder
        thr = df.thr_fit.unique()
        df = pd.concat(objs=[df.loc[(df.thr_fit == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)

        ## Tidy model name and column names
        df = df.rename(columns=metric_names)
        if model_labels is not None:
            df.model_name = df.model_name.replace(model_labels)
        df.to_csv(save_path / out_file, index=False)
    #endregion

    # ---- 6. Net benefit ----
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

    # Tidy model name and column names
    if model_labels is not None:
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

    # ---- 7. Test reduction at sens ----
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
    df = df[['sens', 'model_name', 'proportion_reduction_tests', 'ppv_mod', 'ppv_fit', 'delta_ppv']]

    df = df.loc[df.sens.isin(df0.sens)]

    ## Reorder rows
    thr = df.sens.unique()
    df = pd.concat(objs=[df.loc[(df.sens == t) & (df.model_name == m)] for t in thr for m in model_order if m in df.model_name.unique()], axis=0)

    ## Tidy model name and column names
    df.model_name = df.model_name.replace(model_labels)
    df = df.rename(columns={'thr': 'Threshold approx (%)',
                            'model_name': 'Model',
                            'ppv_mod': 'PPV model (%)',
                            'ppv_fit': 'PPV FIT (%)',
                            'delta_ppv': 'Delta PPV (% scale)',
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
    