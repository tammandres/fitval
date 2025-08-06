"""Plots that combine time periods: time-cut analysis"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PROJECT_ROOT
from pathlib import Path
from fitval.boot import SENS_FIT_CI, SENS_FIT_GAIN_CI, SENS_FIT_2_CI, SENS_FIT_GAIN_2_CI
from fitval.plots import model_labels
from fitval.models import model_colors


model_plot = 'nottingham-cox'  # Model to plot


periods = ['precovid', 'covid', 'post1', 'post2', 'post3', 'post4']

add_method1 = False

fig, ax = plt.subplots(figsize=(12, 6))

for fu in [180, 365]:

    # Labels and colors
    model_label = model_labels[model_plot] #+ '\n(at threshold yielding sensitivity of FIT ≥ 10 µg/g)'

    time_label_plot = {
                'precovid': 'Pre-COVID\n(2017/01 - 2020/02)',
                'covid': 'COVID\n(2020/03 - 2021/04)',
                'post1': 'Post-COVID\n(2021/05 - 2021/12)',
                'post2': '2022 H1\n(2022/01 - 2022/06)',
                'post3': '2022 H2\n(2022/07 - 2022/12)',
                'post4': '2023 H1\n(2023/01 - 2023/06)'
                }
    time_label_plot = {key: val for key, val in time_label_plot.items() if key in periods}

    color = 'C0' if fu == 180 else 'C1'

    save_path = PROJECT_ROOT / 'results' / 'agg' / ('timecut_fu-' + str(fu) + '_agg')

    # Get number of patients and number of CRC cases
    df = pd.read_csv(save_path / SENS_FIT_CI)
    df = df.loc[df.thr_fit == 10]
    dfsub = df.loc[df.metric_name.isin(['pp', 'pn', 'tp', 'fn']) & (df.model_name == 'fit') & (df.model == 'fit')]
    dfsub = dfsub.pivot(index='period', values='metric_value', columns='metric_name')
    print(dfsub)
    n = dfsub.pn + dfsub.pp
    ncrc = dfsub.tp + dfsub.fn
    nfit10 = dfsub.pp
    
    n = pd.concat(objs=[n, ncrc], axis=1)
    n.columns = ['n', 'ncrc']
    n = n.reset_index()

    # Read gain data compared to FIT, method 1
    df = pd.read_csv(save_path / SENS_FIT_GAIN_CI)

    mask = df.metric_name.isin(['precision_gain', 'proportion_reduction_tests',
                                'ppv_mod', 'ppv_fit', 'sens_mod', 'sens_fit', 'delta_sens', 'thr'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   # Rescale to %
    df = df.loc[df.thr_fit == 10]
    df = df.loc[df.metric_name.isin(['proportion_reduction_tests', 'pp_mod', 'pp_fit', 'sens_fit', 'sens_mod', 'delta_sens', 'thr'])]
    df = df.loc[df.period.isin(periods)]

    dfgain = pd.DataFrame()  # Reorder according to period (essential!)
    for p in periods:
        if p in df.period.unique():
            d = df.loc[df.period == p]
            dfgain = pd.concat(objs=[dfgain, d], axis=0)
    dfgain = dfgain.merge(n, how='left')

    # Read gain data compared to FIT, method 2
    df = pd.read_csv(save_path / SENS_FIT_GAIN_2_CI)

    mask = df.metric_name.isin(['precision_gain', 'proportion_reduction_tests', 'delta_sens', 'thr',
                                'ppv_mod', 'ppv_fit', 'sens_mod', 'sens_fit'])
    df.loc[mask, ['metric_value', 'ci_low', 'ci_high']] *= 100   # Rescale to %
    df = df.loc[df.thr_fit == 10]
    df = df.loc[df.metric_name.isin(['proportion_reduction_tests', 'pp_mod', 'pp_fit', 'sens_fit', 'sens_mod', 'delta_sens', 'thr'])]
    df = df.loc[df.period.isin(periods)]

    dfgain2 = pd.DataFrame()  # Reorder according to period (essential!)
    for p in periods:
        if p in df.period.unique():
            d = df.loc[df.period == p]
            dfgain2 = pd.concat(objs=[dfgain2, d], axis=0)
    dfgain2 = dfgain2.merge(n, how='left')

    # Dbl check that point estimates under method 1 and 2 are the same
    m1 = dfgain[['thr_fit', 'model_name', 'metric_name', 'metric_value', 'period']]
    m2 = dfgain[['thr_fit', 'model_name', 'metric_name', 'metric_value', 'period']].rename(columns={'metric_value': 'metric_value_2'})
    m = m1.merge(m2, how='outer', on=['thr_fit', 'model_name', 'metric_name', 'period'])
    test = m.metric_value == m.metric_value_2
    assert test.all()

    # .... Plot percent reduction ....

    ## Method 1 data
    data = dfgain.loc[(dfgain.model_name == model_plot) & (dfgain.metric_name=='proportion_reduction_tests')]
    data['add_low'] = data.ci_low - data.metric_value
    data['add_high'] = data.ci_high - data.metric_value
    data.period = data.period.replace(time_label_plot)
    ci = data[['add_low', 'add_high']].transpose().abs().to_numpy()

    ## Method 2 data
    data2 = dfgain2.loc[(dfgain2.model_name == model_plot) & (dfgain2.metric_name=='proportion_reduction_tests')]
    data2['add_low'] = data2.ci_low - data2.metric_value
    data2['add_high'] = data2.ci_high - data2.metric_value
    data2.period = data2.period.replace(time_label_plot)
    ci2 = data2[['add_low', 'add_high']].transpose().abs().to_numpy()
    #delta = dfgain2.loc[(dfgain2.model_name == model_plot) & (dfgain2.metric_name=='delta_sens')]

    ## Plot
    ax.plot(data2.period, data2.metric_value.to_numpy(), color=color, label=model_label + ", " + str(fu) + "-day follow-up", alpha=0.75)
    ax.scatter(data2.period, data2.metric_value.to_numpy(), color=color, s=64, alpha=0.75)

    ### CI from method 1
    if add_method1:
        #ax.fill_between(x=data.period, y1=data.ci_low, y2=data.ci_high, alpha=0.15, facecolor='black')
        ax.plot(data.period, data.ci_low.to_numpy(), color=color, linestyle='dashed', alpha=0.75)#, label="Confidence interval method 1, fu " + str(fu))
        ax.plot(data.period, data.ci_high.to_numpy(), color=color, linestyle='dashed', alpha=0.75)

    ### CI from method 2
    ax.fill_between(x=data2.period, y1=data2.ci_low, y2=data2.ci_high, alpha=0.15)#, label="Confidence interval method 2, fu " + str(fu))
    ax.errorbar(data2.period, data2.metric_value, yerr=ci2, color=color)

    ax.hlines(y=0, xmin=0, xmax=len(periods)-1, linestyle='dotted', color='red')
    ax.grid(axis='y', linestyle='dotted')
    if not add_method1:
        ax.set(ylim=(-45, 45), ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
        ax.set_yticks(np.arange(-40, 50, 10))
    else:
        ax.set(ylim=(-80, 80), ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
        ax.set_yticks(np.arange(-75, 80, 10))
    ax.legend(frameon=False, loc='upper center')

if add_method1:
    fname = "plot_test_reduction_followup-both_ci-both"
else:
    fname = "plot_test_reduction_followup-both"
suf = '_'  + model_plot
fname += suf + '.png'

plt.savefig(PROJECT_ROOT / 'results' / 'agg' / fname, dpi=300, facecolor='white',  bbox_inches='tight')
plt.close()
