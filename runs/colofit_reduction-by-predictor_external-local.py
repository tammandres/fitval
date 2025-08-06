"""Compute core performance metrics for Nottingham-Cox model,
separating the contributions of different variables.
"""
import pandas as pd
from pathlib import Path
from constants import PROJECT_ROOT
from fitval.boot import _plot_boot
from fitval.models import get_model, model_labels, model_colors
from fitval.metrics import core_metric_at_threshold, metric_at_fit_sens
from fitval.reformat import _reformat_ci
import numpy as np
import matplotlib.pyplot as plt


# Settings
fu = 180
B = 1000
seed = 42

out_path = PROJECT_ROOT / 'results' / 'reduction_thr-external-local'
out_path.mkdir(exist_ok=True, parents=True)


# ---- Read data and divide to time periods
#region
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
fitsub = fit.loc[(fit.stool_pot == 0) & (fit.fit_date >= date_thr)]

post3buff = fitsub.loc[(fitsub.fit_date >= '2022-07-01') & (fitsub.fit_date < '2023-01-01')]
post4buff = fitsub.loc[(fitsub.fit_date >= '2023-01-01') & (fitsub.fit_date < '2023-07-01')]

fit_data = {'all': fit,
            'precovid': precovid, 
            'covid': covid, 
            'post1': post1,
            'post2': post2, 
            'post3': post3,
            'post4': post4,
            #'post3-buffcomm': post3buff,
            #'post4-buffcomm': post4buff,
            }

if fu == 365:  # too small n
    fit_data.pop('post4')
    fit_data.pop('post4-buffcomm')

cox_model = get_model('nottingham-cox')

matrix = {}
for label, f in fit_data.items():
    dfsub = df.loc[df.patient_id.isin(f.patient_id)].copy()

    # Add model predictions to data matrix
    dfsub['y_pred'] = cox_model(dfsub)
    pred = cox_model(dfsub, return_fx=True)[1].copy()
    pred.index = dfsub.index
    pred.columns = ['pred_' + c for c in pred.columns]
    pred['base'] = np.exp(-0.6592014)
    dfsub = pd.concat(objs=[dfsub, pred], axis=1)

    matrix[label] = dfsub
    print(label, dfsub.shape[0], dfsub.crc.sum())

#endregion


# ---- Compute metrics and save to disk
#region

def get_predictor(df, predictor):
    if predictor == 'fit_mcv':
        y_pred = df.pred_fit_val + df.pred_blood_MCV
    elif predictor == 'fit_mcv_plt':
        y_pred = df.pred_fit_val + df.pred_blood_MCV + df.pred_blood_PLT
    elif predictor == 'fit_plt':
        y_pred = df.pred_fit_val + df.pred_blood_PLT
    elif predictor == 'fit_age_sex':
        y_pred = df.pred_fit_val + df.pred_age_at_fit + df.pred_ind_gender_M
    elif predictor == 'fit_age':
        y_pred = df.pred_fit_val + df.pred_age_at_fit
    elif predictor == 'fit_age_mcv':
        y_pred = df.pred_fit_val + df.pred_age_at_fit + df.pred_blood_MCV
    elif predictor == 'fit_sex_mcv':
        y_pred = df.pred_fit_val + df.pred_age_at_fit + df.pred_blood_MCV
    elif predictor == 'full':
        y_pred = cox_model(df)
    
    if predictor != 'full':
        y_pred = y_pred.to_numpy()
    return y_pred

gain = pd.DataFrame()
gain_boot = pd.DataFrame()

rng = np.random.default_rng(seed=seed)
labels = list(matrix.keys())
predictors = ['fit_mcv', 'fit_mcv_plt', 'fit_age_sex', 'fit_age', 'full']

for i, label in enumerate(labels):
    print('\n----', label)

    # Current and previous data subsets
    dfsub = matrix[label]
    y_true, fit_val = dfsub.crc.to_numpy(), dfsub.fit_val.to_numpy()
    if i > 1:
        label_prev = labels[i - 1]
        df_prev = matrix[label_prev]
        y_prev, fit_prev = df_prev.crc.to_numpy(), df_prev.fit_val.to_numpy()

    # Metrics on original data for all predictors
    for predictor in predictors:

        # Estimate model threshold on previous data subset
        if i > 1:
            pred_prev = get_predictor(df_prev, predictor)
            m, __ = metric_at_fit_sens(y_prev, pred_prev, fit_prev, thr_fit = [10])
            thr_mod = m.loc[m.model == 'model', 'thr'].item()
        else:
            thr_mod = None

        # Performance on original data
        y_pred = get_predictor(dfsub, predictor)
        
        ## Local threshold using current data subset
        m_curr, __ = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit = [10])
        thr_mod_curr = m_curr.loc[m_curr.model == 'model', 'thr'].item()
        g_curr = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_mod_curr, thr_fit = 10, long_format=False)
        g_curr = pd.melt(g_curr, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
        g_curr['thr_method'] = 'local-current'
        g = g_curr

        ## Local threshold using previous data subset
        if thr_mod is not None:
            g2 = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_mod, thr_fit = 10, long_format=False)
            g2 = pd.melt(g2, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
            g2['b'] = -1
            g2['thr_method'] = 'local-previous'
            g = pd.concat(objs=[g, g2], axis = 0)

        g['predictor'] = predictor
        g['b'] = -1

        # Metrics on bootstrap samples for all predictors
        g_boot = pd.DataFrame()
        for b in range(B):
            print(f"\r{b}", end="", flush=True)

            # Get bootstrap sample
            idx_boot = rng.choice(a=np.arange(len(y_true)), size=len(y_true), replace=True)
            y_boot, fit_boot, pred_boot = y_true[idx_boot], fit_val[idx_boot], y_pred[idx_boot]

            gb_curr = core_metric_at_threshold(y_boot, pred_boot, fit_boot, thr_mod = thr_mod_curr, thr_fit = 10, long_format=False)
            gb_curr = pd.melt(gb_curr, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
            gb_curr['thr_method'] = 'local-current'
            gb = gb_curr

            if thr_mod is not None:
                gb2 = core_metric_at_threshold(y_boot, pred_boot, fit_boot, thr_mod = thr_mod, thr_fit = 10, long_format=False)
                gb2 = pd.melt(gb2, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
                gb2['thr_method'] = 'local-previous'
                gb = pd.concat(objs=[gb, gb2], axis = 0)
        
            gb['b'] = b
            gb['predictor'] = predictor
            g_boot = pd.concat(objs=[g_boot, gb], axis=0)
                
        q = g_boot.groupby(['thr_method', 'thr_mod', 'thr_fit', 'metric_name']).metric_value.agg([lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
        q.columns = ['ci_low', 'ci_high']
        q = q.reset_index()
        g = g.merge(q, how='left')

        g['period'] = label
        g_boot['period'] = label

        # Store
        gain = pd.concat(objs=[gain, g], axis=0)
        gain_boot = pd.concat(objs=[gain_boot, g_boot], axis=0)

gain.to_csv(out_path / ('model-cox-by-predictor_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'), index=False)
#endregion


# ---- Visualise bootstrap distributions
#region
visualise_boot = True
if visualise_boot:

    df = pd.concat(objs=[gain, gain_boot], axis=0)
    df = df.drop(labels=['ci_low', 'ci_high'], axis=1)
    df = df.loc[~df.metric_name.isin(['pp_mod_1000', 'pp_fit_1000'])]
    df = df.drop(labels=['thr_fit', 'thr_mod'], axis=1)

    for predictor in predictors:
        print(predictor)

        model_colors[predictor] = 'C0'

        df2 = df.loc[df.thr_method == 'local-previous'].drop(labels=['thr_method'], axis=1)
        df2.period.unique()
        df2 = df2.loc[df2.predictor == predictor].drop(labels=['predictor'], axis=1)
        df2['model_name'] = predictor
        _plot_boot(df2, out_path, 'plot-boot_model-cox-' + predictor + '_thr-local-prev_fu-' + str(fu) + '_b-' + str(B) + '.png', sample=False, model_colors=model_colors)

        df3 = df.loc[df.thr_method == 'local-current'].drop(labels=['thr_method'], axis=1)
        df3.period.unique()
        df3 = df3.loc[df3.predictor == predictor].drop(labels=['predictor'], axis=1)
        df3['model_name'] = predictor
        _plot_boot(df3, out_path, 'plot-boot_model-cox-' + predictor + '_thr-local-curr_fu-' + str(fu) + '_b-' + str(B) + '.png', sample=False, model_colors=model_colors)
#endregion


# ---- Reformat metrics
#region

fname = 'model-cox-by-predictor_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'
gain = pd.read_csv(out_path / fname)

df = gain.copy()
df['q025'] = df.ci_low
df['q975'] = df.ci_high

## Rescale to %
mask = df.metric_name.isin(['precision_gain', 'proportion_reduction_tests', 'delta_sens',
                            'sens_mod', 'sens_fit', 'prevalence'])
df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   

## To wide format
df = _reformat_ci(df, digits=2)
df = df.drop_duplicates()
df = df.pivot(index=['period', 'thr_method', 'predictor', 'thr_mod', 'thr_fit'], columns='metric_name', values='metric_value').reset_index()
df = df[['period', 'predictor', 'thr_method', 'prevalence', 'proportion_reduction_tests', 'pp_mod_1000', 'pp_fit_1000',
         'delta_sens', 'sens_mod', 'sens_fit', 'thr_mod', 'thr_fit', 'pp_mod', 'pp_fit'
         ]]

## Tidy model name and period names
df.model_name = df.model_name.replace(model_labels)

time_labels = {
                'precovid': 'Pre-COVID (2017/01 - 2020/02)',
                'covid': 'COVID (2020/03 - 2021/04)',
                'post1': 'Post-COVID (2021/05 - 2021/12)',
                'post2': '2022 H1 (2022/01 - 2022/06)',
                'post3': '2022 H2 (2022/07 - 2022/12)',
                'post4': '2023 H1 (2023/01 - 2023/06)'
                }

if fu == 365:
    time_labels['all'] = 'All data (2017/01 - 2023/02)'
else:
    time_labels['all'] = 'All data (2017/01 - 2023/08)'

df = df.set_index('period').loc[['all', 'precovid', 'covid', 'post1', 'post2', 'post3', 'post4']].reset_index()

df.period = df.period.replace(time_labels)

## Rename columns
df = df.rename(columns={'thr_fit': 'FIT threshold (ug/g)', 
                        'thr': 'Model threshold (%)', 
                        'thr_mod': 'Model threshold (%)',
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
                        'sens_fit': 'Sensitivity FIT (%)',
                        'period': 'Period',
                        'prevalence': 'Prevalence',
                        'thr_method': 'Threshold method',
                        'predictor': 'Predictor'})


df.to_csv(out_path / ('reformat_model-cox-by-predictor_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'), index=False)


df[['Period', 'Predictor', 'Threshold method', 'Percent reduction in number of positive tests',
    'Positive tests per 1000 tests (model)', 'Positive tests per 1000 tests (FIT)',
    'Delta sensitivity (model minus FIT)', 'Sensitivity model (%)', 'Sensitivity FIT (%)']]

dfsub = df.loc[df['Threshold method'] == 'local-current']
dfsub[['Period', 'Predictor', 'Percent reduction in number of positive tests', 'Delta sensitivity (model minus FIT)', 'Model threshold (%)']]



#endregion


# ---- Plot
#region

# Prepare data
time_labels = { 
                'precovid': 'Pre-COVID\n(2017/01 - 2020/02)',
                'covid': 'COVID\n(2020/03 - 2021/04)',
                'post1': 'Post-COVID\n(2021/05 - 2021/12)',
                'post2': '2022 H1\n(2022/01 - 2022/06)',
                'post3': '2022 H2\n(2022/07 - 2022/12)',
                'post4': '2023 H1\n(2023/01 - 2023/06)'
                }
if fu == 365:
    time_labels['all'] = 'All data\n(2017/01 - 2023/02)'
else:
    time_labels['all'] = 'All data\n(2017/01 - 2023/08)'

thr_method_labels = {'external': 'Externally derived threshold',
                     'local-current': 'Locally derived threshold\n(current data)',
                     'local-previous': 'Locally derived threshold\n(previous data subset)'}

fname = 'model-cox-by-predictor_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'
gain = pd.read_csv(out_path / fname)

data = gain.loc[(gain.metric_name=='proportion_reduction_tests') & (gain.thr_method=='local-current')].copy()
data[['metric_value', 'ci_low', 'ci_high']] *= 100
data['add_low'] = data.ci_low - data.metric_value
data['add_high'] = data.ci_high - data.metric_value
data = data.loc[data.period != 'all']
data.period = data.period.replace(time_labels)

data2 = gain.loc[(gain.metric_name=='proportion_reduction_tests') & (gain.thr_method=='local-previous')].copy()
data2[['metric_value', 'ci_low', 'ci_high']] *= 100
data2['add_low'] = data2.ci_low - data2.metric_value
data2['add_high'] = data2.ci_high - data2.metric_value
data2 = data2.loc[data2.period != 'all']
data2.period = data2.period.replace(time_labels)

data3 = gain.loc[(gain.metric_name=='delta_sens') & (gain.thr_method=='local-current')].copy()
data3[['metric_value', 'ci_low', 'ci_high']] *= 100
data3['add_low'] = data3.ci_low - data3.metric_value
data3['add_high'] = data3.ci_high - data3.metric_value
data3 = data3.loc[data3.period != 'all']
data3.period = data3.period.replace(time_labels)

periods = data.period.drop_duplicates()
xmap = {period:i for i, period in enumerate(periods)}
xmap_inv = {i:period for period, i in xmap.items()}
x = np.arange(len(data.period.drop_duplicates()))
bar_width = 0.075

# Plot
markersize = 14
model_label = 'Nottingham-Cox'
fig, ax = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True)

thr_methods = ['local-current', 'local-previous']
predictors = ['full', 'fit_age', 'fit_age_sex', 'fit_mcv', 'fit_mcv_plt']
predictor_colors = {'full': 'C0',
                    'fit_age': 'C1',
                    'fit_age_sex': 'C2',
                    'fit_mcv': 'C3',
                    'fit_mcv_plt': 'C4'}
predictor_labels = {'full': 'Full model (FIT, age, sex, MCV, PLT)',
                    'fit_mcv': 'FIT, MCV',
                    'fit_mcv_plt': 'FIT, MCV, PLT',
                    'fit_age': 'FIT, age',
                    'fit_age_sex': 'FIT, age, sex'}

show_line = True
plot_sens = False
for i, predictor in enumerate(predictors):

    # Get data subset for current thr_method
    datasub = data.loc[(data.predictor == predictor)]
    ci = datasub[['add_low', 'add_high']].transpose().abs().to_numpy()

    datasub2 = data2.loc[(data2.predictor == predictor)]
    ci2 = datasub2[['add_low', 'add_high']].transpose().abs().to_numpy()

    datasub3 = data3.loc[(data3.predictor == predictor)]
    ci3 = datasub3[['add_low', 'add_high']].transpose().abs().to_numpy()

    # Colors and some labels
    color = predictor_colors[predictor]
    label = predictor_labels[predictor]
    axes_color = ax[0].spines['bottom'].get_edgecolor()
    axes_width = ax[0].spines['bottom'].get_linewidth()

    # Plot reduction in referrals: local-curr
    if show_line:
        ax[0].plot(datasub.period.replace(xmap) + i * bar_width, 
                datasub.metric_value.to_numpy(), 
                color=color, label=label, alpha=1)
        ax[0].fill_between(x=datasub.period.replace(xmap) + i * bar_width, 
                        y1=datasub.ci_low, y2=datasub.ci_high, alpha=0.15, color=color) 
        err_label = None
    else:
        err_label = label
    ax[0].errorbar(datasub.period.replace(xmap) + i * bar_width, 
                datasub.metric_value, 
                yerr=ci, color=color, fmt='.', markersize=markersize, alpha=1, label=err_label)
    ax[0].axhline(y=0, color='red', linestyle='dotted')
    
    # Plot reduction in referrals: local-prev
    if plot_sens:
        ax[1].plot(datasub3.period.replace(xmap) + i * bar_width, 
                   datasub3.metric_value.to_numpy(), 
                   color=color, label=label, alpha=0.75)
        ax[1].fill_between(x=datasub3.period.replace(xmap) + i * bar_width, 
                           y1=datasub3.ci_low, y2=datasub3.ci_high, alpha=0.15) 
        ax[1].errorbar(datasub3.period.replace(xmap) + i * bar_width, 
                       datasub3.metric_value, 
                       yerr=ci3, color=color, fmt='.', markersize=markersize, alpha=0.75)
    else:
        if show_line:
            ax[1].plot(datasub2.period.replace(xmap) + i * bar_width, 
                    datasub2.metric_value.to_numpy(), 
                    color=color, label=label, alpha=0.75)
            ax[1].fill_between(x=datasub2.period.replace(xmap) + i * bar_width, 
                            y1=datasub2.ci_low, y2=datasub2.ci_high, alpha=0.15, color=color)
            err_label = None
        else:
            err_label = label
        ax[1].errorbar(datasub2.period.replace(xmap) + i * bar_width, 
                    datasub2.metric_value, 
                    yerr=ci2, color=color, fmt='.', markersize=markersize, alpha=0.75, label=err_label)
        ax[1].axhline(y=0, color='red', linestyle='dotted')
    
ax[0].grid(axis='y', linestyle='dotted')
ax[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
ax[1].grid(axis='y', linestyle='dotted')
ax[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')

if plot_sens:
    ax[0].set(title='A. Reduction in referrals',
              ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
    ax[1].set(title='B. Percent cancers missed',
              ylabel="Percent cancers missed\ncompared to FIT ≥ 10 µg/g (negative is worse)")
else:
    ax[0].set(title='A. Threshold estimated locally on current time period',
          ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
    ax[1].set(title='B. Threshold estimated locally on previous time period',
            ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")

#ax[0].set_xticks(x)
ax[0].set_xticks(x + bar_width * (len(predictors) - 1) / 2)
ax[0].set_xticklabels(periods)
#ax[0].set_yticks(np.arange(-40, 15, 5))

#ax[1].set_xticks(x)
ax[1].set_xticks(x + bar_width * (len(predictors) - 1) / 2)
ax[1].set_xticklabels(periods)
ax[1].set_xlim(ax[0].get_xlim())

if show_line:
    fname = "plot_reduction_model-cox-by-predictor_thr-external-local_fu-" + str(fu) + "_b-" + str(B) + '.png'
else:
    fname = "plot_reduction_model-cox-by-predictor_thr-external-local_fu-" + str(fu) + "_b-" + str(B) + '_noline.png'
plt.savefig(out_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
plt.close()


#endregion