"""Compute core performance metrics for Nottingham-Cox model compared to FIT test, 
using externally estimated 0.6% risk threshold,
or locally estimated threshold using current data
or locally estimated threshold using previous data subset
"""
import pandas as pd
from pathlib import Path
from constants import PROJECT_ROOT
from fitval.boot import _plot_boot
from fitval.models import get_model, model_labels
from fitval.metrics import core_metric_at_threshold, metric_at_fit_sens
from fitval.reformat import _reformat_ci
import numpy as np
import matplotlib.pyplot as plt


# Settings
thr_colofit = 0.0064
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

matrix = {}
for label, f in fit_data.items():
    dfsub = df.loc[df.patient_id.isin(f.patient_id)]
    matrix[label] = dfsub
    print(label, dfsub.shape[0], dfsub.crc.sum())
#endregion


# ---- Compute metrics and save to disk
#region
def eval_rule_out(y_true, y_pred, fit_val, thr_mod, digits=2):

    # Number FIT positives ruled out, number FIT negatives ruled in
    pos_mod = y_pred >= thr_mod
    pos_fit = fit_val >= 10
    reduction = round((pos_mod.sum() - pos_fit.sum()) / pos_fit.sum() * 100, digits)
    n_rule_out = pos_fit[~pos_mod].sum()
    n_rule_in = (~pos_fit)[pos_mod].sum()
    p_rule_out = round(n_rule_out / (n_rule_out + n_rule_in) * 100, digits)
    ratio_out_in = round(n_rule_out / n_rule_in, digits)

    reduction_rule_out = round((pos_fit[pos_mod].sum() - pos_fit.sum()) / pos_fit.sum() * 100, digits)

    # Number FIT positive cancers ruled out, number FIT negative cancers ruled in
    crc_fit_pos = y_true[pos_fit].sum()
    crc_fit_pos_out = y_true[pos_fit & (~pos_mod)].sum()
    crc_fit_neg = y_true[~pos_fit].sum()
    crc_fit_neg_in = y_true[~pos_fit & pos_mod].sum()
    p_crc_mis_out = round(crc_fit_pos_out / crc_fit_pos * 100, digits)
    data = {'p_reduction': reduction, 'p_reduction_rule_out': reduction_rule_out,
            'n_fit_pos': pos_fit.sum(), 
            'n_rule_out': n_rule_out, 'n_rule_in': n_rule_in, 'ratio_out_in': ratio_out_in, 
            'p_rule_out': p_rule_out,
            'crc_fit_pos': crc_fit_pos, 'crc_fit_pos_out': crc_fit_pos_out,
            'crc_fit_neg': crc_fit_neg, 'crc_fit_neg_in': crc_fit_neg_in,
            'p_crc_fit_pos_out': p_crc_mis_out,
            'thr_mod_p': thr_mod * 100
            }
    data = pd.DataFrame(data, index=[0])
    return data


model_name = 'nottingham-cox'
model = get_model(model_name)

gain = pd.DataFrame()
gain_boot = pd.DataFrame()
rule_out = pd.DataFrame()

rng = np.random.default_rng(seed=seed)
labels = list(matrix.keys())

for i, label in enumerate(labels):
    print('\n----', label, matrix[label].shape)

    # Estimate model threshold on previous data subset
    if i > 1:
        label_prev = labels[i - 1]
        df_prev = matrix[label_prev]
        y_prev, fit_prev = df_prev.crc.to_numpy(), df_prev.fit_val.to_numpy()
        pred_prev = model(df_prev)
        m, __ = metric_at_fit_sens(y_prev, pred_prev, fit_prev, thr_fit = [10])
        thr_mod = m.loc[m.model == 'model', 'thr'].item()
    else:
        thr_mod = None

    # Performance on original data
    dfsub = matrix[label]
    y_true, fit_val = dfsub.crc.to_numpy(), dfsub.fit_val.to_numpy()
    y_pred = model(dfsub)

    ## External threshold
    g = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_colofit, thr_fit = 10, long_format=False)
    g = pd.melt(g, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
    g['thr_method'] = 'external'

    ## Check whether reduction in referrals comes from ruling out FIT positives
    r = eval_rule_out(y_true, y_pred, fit_val, thr_colofit)
    r['thr_method'] = 'external'
    r['period'] = label
    rule_out = pd.concat(objs=[rule_out, r], axis=0)
    print('Rule out external:\n', r)

    ## Local threshold using current data subset
    m_curr, __ = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit = [10])
    thr_mod_curr = m_curr.loc[m_curr.model == 'model', 'thr'].item()
    g_curr = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_mod_curr, thr_fit = 10, long_format=False)
    g_curr = pd.melt(g_curr, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
    g_curr['thr_method'] = 'local-current'
    g = pd.concat(objs=[g, g_curr], axis=0)

    ## Check whether reduction in referrals comes from ruling out FIT positives
    r = eval_rule_out(y_true, y_pred, fit_val, thr_mod_curr)
    r['thr_method'] = 'local-current'
    r['period'] = label
    rule_out = pd.concat(objs=[rule_out, r], axis=0)
    print('Rule out current:\n', r)

    ## Local threshold using previous data subset
    if thr_mod is not None:
        g2 = core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = thr_mod, thr_fit = 10, long_format=False)
        g2 = pd.melt(g2, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
        g2['b'] = -1
        g2['thr_method'] = 'local-previous'
        g = pd.concat(objs=[g, g2], axis = 0)
    
        ## Check whether reduction in referrals comes from ruling out FIT positives
        r = eval_rule_out(y_true, y_pred, fit_val, thr_mod)
        r['thr_method'] = 'local-previous'
        r['period'] = label
        rule_out = pd.concat(objs=[rule_out, r], axis=0)
        print('Rule out previous:\n', r)

    g['b'] = -1

    # Bootstrap
    g_boot = pd.DataFrame()
    for b in range(B):
        print(f"\r{b}", end="", flush=True)

        idx_boot = rng.choice(a=np.arange(len(y_true)), size=len(y_true), replace=True)
        y_boot, pred_boot, fit_boot = y_true[idx_boot], y_pred[idx_boot], fit_val[idx_boot]

        gb = core_metric_at_threshold(y_boot, pred_boot, fit_boot, thr_mod = thr_colofit, thr_fit = 10, long_format=False)
        gb = pd.melt(gb, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
        gb['thr_method'] = 'external'

        gb_curr = core_metric_at_threshold(y_boot, pred_boot, fit_boot, thr_mod = thr_mod_curr, thr_fit = 10, long_format=False)
        gb_curr = pd.melt(gb_curr, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
        gb_curr['thr_method'] = 'local-current'
        gb = pd.concat(objs=[gb, gb_curr], axis=0)

        if thr_mod is not None:
            gb2 = core_metric_at_threshold(y_boot, pred_boot, fit_boot, thr_mod = thr_mod, thr_fit = 10, long_format=False)
            gb2 = pd.melt(gb2, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
            gb2['thr_method'] = 'local-previous'
            gb = pd.concat(objs=[gb, gb2], axis = 0)
        
        gb['b'] = b
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

gain.to_csv(out_path / ('model-cox_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'), index=False)
rule_out.to_csv(out_path / ('model-cox_thr-external-local_fu-' + str(fu) + '_ruleout.csv'), index=False)
#endregion


# ---- Visualise bootstrap distributions
#region
df = pd.concat(objs=[gain, gain_boot], axis=0)
df = df.drop(labels=['ci_low', 'ci_high'], axis=1)
df = df.loc[~df.metric_name.isin(['pp_mod_1000', 'pp_fit_1000'])]
df = df.drop(labels=['thr_fit', 'thr_mod'], axis=1)
df['model_name'] = 'nottingham-cox'

df1 = df.loc[df.thr_method == 'external'].drop(labels=['thr_method'], axis=1)
_plot_boot(df1, out_path, 'plot-boot_model-cox_thr-0.0064_fu-' + str(fu) + '_b-' + str(B) + '.png', sample=False)

df2 = df.loc[df.thr_method == 'local-previous'].drop(labels=['thr_method'], axis=1)
df2.period.unique()
_plot_boot(df2, out_path, 'plot-boot_model-cox_thr-local-prev_fu-' + str(fu) + '_b-' + str(B) + '.png', sample=False)

df3 = df.loc[df.thr_method == 'local-current'].drop(labels=['thr_method'], axis=1)
df3.period.unique()
_plot_boot(df3, out_path, 'plot-boot_model-cox_thr-local-curr_fu-' + str(fu) + '_b-' + str(B) + '.png', sample=False)
#endregion


# ---- Reformat metrics
#region

fname = 'model-cox_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'
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
df = df.pivot(index=['period', 'thr_method', 'thr_mod', 'thr_fit'], columns='metric_name', values='metric_value').reset_index()
df = df[['period', 'thr_method', 'prevalence', 'proportion_reduction_tests', 'pp_mod_1000', 'pp_fit_1000',
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
                        'thr_method': 'Threshold method'})


df.to_csv(out_path / ('reformat_model-cox_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'), index=False)


df[['Period', 'Threshold method', 'Percent reduction in number of positive tests',
    'Positive tests per 1000 tests (model)', 'Positive tests per 1000 tests (FIT)',
    'Delta sensitivity (model minus FIT)', 'Sensitivity model (%)', 'Sensitivity FIT (%)']]

dfsub = df.loc[df['Threshold method'] == 'external']
dfsub = dfsub[['Period', 'Percent reduction in number of positive tests', 'Positive tests per 1000 tests (model)',
                'Positive tests per 1000 tests (FIT)', 'Delta sensitivity (model minus FIT)', 'Model threshold (%)']]
dfsub.columns = ['period', 'red', 'posmod', 'posfit', 'delta', 'thr']
dfsub

#endregion


# ---- Plot
#region

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

fname = 'model-cox_thr-external-local_fu-' + str(fu) + '_b-' + str(B) + '_gain2.csv'
gain = pd.read_csv(out_path / fname)

data = gain.loc[(gain.metric_name=='proportion_reduction_tests')].copy()
data[['metric_value', 'ci_low', 'ci_high']] *= 100
data['add_low'] = data.ci_low - data.metric_value
data['add_high'] = data.ci_high - data.metric_value
data.period = data.period.replace(time_labels)

data2 = gain.loc[(gain.metric_name=='delta_sens')].copy()
data2[['metric_value', 'ci_low', 'ci_high']] *= 100
data2['add_low'] = data2.ci_low - data2.metric_value
data2['add_high'] = data2.ci_high - data2.metric_value
data2.period = data2.period.replace(time_labels)

periods = data.period.drop_duplicates()
xmap = {period:i for i, period in enumerate(periods)}
xmap_inv = {i:period for period, i in xmap.items()}
x = np.arange(len(data.period.drop_duplicates()))
bar_width = 0.075

# Plot
markersize = 14
model_label = 'Nottingham-Cox'
fig, ax = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True)

thr_methods = ['local-current', 'local-previous', 'external']
#thr_methods = ['local-current', 'local-previous']
#thr_methods = ['external']
#thr_methods = ['local-current']
for i, thr_method in enumerate(thr_methods):
    print(thr_method)

    # Get data subset for current thr_method
    datasub = data.loc[data.thr_method == thr_method]
    ci = datasub[['add_low', 'add_high']].transpose().abs().to_numpy()

    datasub2 = data2.loc[data2.thr_method == thr_method]
    ci2 = datasub2[['add_low', 'add_high']].transpose().abs().to_numpy()

    # Colors and some labels
    color = 'C' + str(i)
    label_thr = thr_method_labels[thr_method]
    axes_color = ax[0].spines['bottom'].get_edgecolor()
    axes_width = ax[0].spines['bottom'].get_linewidth()

    # Plot reduction in referrals
    periods_sub = datasub.period.tolist()
    if any([p.lower().startswith('all data') for p in periods_sub]):
        ax[0].errorbar(datasub.period.replace(xmap).iloc[0] + i * bar_width, 
                       datasub.metric_value.iloc[0], 
                       yerr=ci[:, 0].reshape(-1, 1), 
                       color=color, fmt='.', markersize=markersize, alpha=1)
        ax[0].axvline(x=0.5, color=axes_color, linestyle='solid', linewidth=axes_width)
        
        ax[0].plot(datasub.period.replace(xmap)[1:] + i * bar_width, 
                   datasub.metric_value.to_numpy()[1:], 
                   color=color, label=label_thr, alpha=1)
        ax[0].fill_between(x=datasub.period.replace(xmap)[1:] + i * bar_width, 
                           y1=datasub.ci_low[1:], y2=datasub.ci_high[1:], alpha=0.15)
        ax[0].errorbar(datasub.period.replace(xmap)[1:] + i * bar_width, 
                       datasub.metric_value[1:], 
                       yerr=ci[:, 1:], color=color, fmt='.', markersize=markersize, alpha=1)
    else:
        ax[0].plot(datasub.period.replace(xmap) + i * bar_width, 
                   datasub.metric_value.to_numpy(), 
                   color=color, label=label_thr, alpha=1)
        ax[0].fill_between(x=datasub.period.replace(xmap) + i * bar_width, 
                           y1=datasub.ci_low, y2=datasub.ci_high, alpha=0.15) 
        ax[0].errorbar(datasub.period.replace(xmap) + i * bar_width, 
                       datasub.metric_value, 
                       yerr=ci, color=color, fmt='.', markersize=markersize, alpha=1)
    ax[0].axhline(y=0, color='red', linestyle='dotted')
    
    # Plot cancers missed
    if any([p.lower().startswith('all data') for p in periods_sub]):
        ax[1].errorbar(datasub2.period.replace(xmap).iloc[0] + i * bar_width, 
                       datasub2.metric_value.iloc[0], 
                       yerr=ci2[:, 0].reshape(-1, 1), color=color, fmt='.', markersize=markersize, alpha=0.75)
        ax[1].axvline(x=0.5, color=axes_color, linestyle='solid', linewidth=axes_width)
        ax[1].plot(datasub2.period.replace(xmap)[1:] + i * bar_width, 
                   datasub2.metric_value.to_numpy()[1:], 
                   color=color, label=label_thr, alpha=0.75)

        ax[1].fill_between(x=datasub2.period.replace(xmap)[1:] + i * bar_width, 
                           y1=datasub2.ci_low[1:], y2=datasub2.ci_high[1:], alpha=0.15)
        ax[1].errorbar(datasub2.period.replace(xmap)[1:] + i * bar_width, 
                       datasub2.metric_value[1:], 
                       yerr=ci2[:, 1:], color=color, fmt='.', markersize=markersize, alpha=0.75)
    else:
        ax[1].plot(datasub2.period.replace(xmap) + i * bar_width, 
                   datasub2.metric_value.to_numpy(), 
                   color=color, label=label_thr, alpha=0.75)
        ax[1].fill_between(x=datasub2.period.replace(xmap) + i * bar_width, 
                           y1=datasub2.ci_low, y2=datasub2.ci_high, alpha=0.15) 
        ax[1].errorbar(datasub2.period.replace(xmap) + i * bar_width, 
                       datasub2.metric_value, 
                       yerr=ci2, color=color, fmt='.', markersize=markersize, alpha=0.75)
    ax[1].axhline(y=0, color='red', linestyle='dotted')
    
ax[0].grid(axis='y', linestyle='dotted')
ax[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
ax[0].set(title='A. Percent reduction in referrals',
          ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
ax[1].grid(axis='y', linestyle='dotted')
ax[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
ax[1].set(title='B. Percent cancers missed',
          ylabel="Percent cancers missed\ncompared to FIT ≥ 10 µg/g (negative is worse)")

ax[0].set_xticks(x)
ax[0].set_xticks(x + bar_width * (len(thr_methods) - 1) / 2)
ax[0].set_xticklabels(periods)
ax[0].set_yticks(np.arange(-40, 15, 5))

ax[1].set_xticks(x)
ax[1].set_xticks(x + bar_width * (len(thr_methods) - 1) / 2)
ax[1].set_xticklabels(periods)


fname = "plot_reduction_model-cox_thr-external-local_fu-" + str(fu) + "_b-" + str(B) + '.png'
#fname = "plot_reduction_model-cox_thr-local_fu-" + str(fu) + "_b-" + str(B) + '.png'
#fname = "plot_reduction_model-cox_thr-external_fu-" + str(fu) + "_b-" + str(B) + '.png'
#fname = "plot_reduction_model-cox_thr-local-curr_fu-" + str(fu) + "_b-" + str(B) + '.png'
plt.savefig(out_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')

fname_svg = "plot_reduction_model-cox_thr-external-local_fu-" + str(fu) + "_b-" + str(B) + '.svg'
plt.savefig(out_path / fname_svg, dpi=300, facecolor='white',  bbox_inches='tight')

plt.close()


#endregion



# So

