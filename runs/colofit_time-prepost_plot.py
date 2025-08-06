"""Plots that combine time periods: time-cut analysis"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import PROJECT_ROOT
from pathlib import Path
from fitval.boot import SENS_FIT_CI, SENS_FIT_GAIN_CI, SENS_FIT_2_CI, SENS_FIT_GAIN_2_CI
from fitval.plots import model_labels, plot_rocpr_agg, plot_cal_smooth_agg, plot_dca_agg
from fitval.models import model_colors


model_plot = 'nottingham-cox'  # Model to plot
periods = ['all', 'prebuff', 'bufftime', 'buffcomm']


# Loop over follow-up times
for fu in [180, 365]:
    
    save_path = PROJECT_ROOT / 'results' / 'agg' / ('prepost_fu-' + str(fu) + '_agg')

    # ---- 1. Plot reduction in number of tests, and number of tests, at sensitivity of FIT >= 10 ----

    # Labels
    #region
    model_label = model_labels[model_plot] + '\n(at threshold yielding sensitivity of FIT ≥ 10 µg/g)'

    if fu == 365:
        time_label_plot = {
                    'all': 'All data\n(2017/01 - 2023/02)',
                    'prebuff': 'Pre-buffer data\n(2017/01 - 2021/06)',
                    'bufftime': 'Buffer data from time\n(2021/10 - 2023/02)',
                    'buffcomm': 'Buffer data from comment\n(2022/04 - 2023/02)',
                    }
    else:
        time_label_plot = {
                    'all': 'All data\n(2017/01 - 2023/08)',
                    'prebuff': 'Pre-buffer data\n(2017/01 - 2021/06)',
                    'bufftime': 'Buffer data from time \n(2021/10 - 2023/08)',
                    'buffcomm': 'Buffer data from comment\n(2022/04 - 2023/08)',
                    }    

    time_label_plot = {key: val for key, val in time_label_plot.items() if key in periods}
    #endregion

    # .... 1.1. Prepare data .... 
    #region

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
    #endregion

    # .... 1.2. Plot percent reduction ....
    #region

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
    for ci_method in ["method2", "both"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        line_plot = False

        if not line_plot:
            ax.bar(data2.period, data2.metric_value.to_numpy(), yerr=ci2, capsize=5, 
                color='C0', label=model_label, # ['C' + str(i) for i in range(len(periods))], 
                edgecolor=None)
            if ci_method == "both":
                ax.errorbar(data2.period, data2.metric_value.to_numpy(), yerr=ci, capsize=0, 
                            alpha = 0.7, marker='None', color='gray',linestyle='None')
        else:
            ax.plot(data2.period, data2.metric_value.to_numpy(), color='C0', label=model_label)
            ax.scatter(data2.period, data2.metric_value.to_numpy(), color='C0', s=48)

            ### CI from method 1
            #ax.fill_between(x=data.period, y1=data.ci_low, y2=data.ci_high, alpha=0.15, facecolor='black')
            ax.plot(data.period, data.ci_low.to_numpy(), color='C0', linestyle='dashed', alpha=0.75, label="Confidence interval method 1")
            ax.plot(data.period, data.ci_high.to_numpy(), color='C0', linestyle='dashed', alpha=0.75)

            ### CI from method 2
            ax.fill_between(x=data2.period, y1=data2.ci_low, y2=data2.ci_high, alpha=0.15, label="Confidence interval method 2")
            ax.errorbar(data2.period, data2.metric_value, yerr=ci2, color='C0')


        ax.hlines(y=0, xmin=-0.5, xmax=len(periods)-1 + 0.5, linestyle='solid', color='red')
        ax.grid(axis='y', linestyle='dotted')
        ax.set(ylim=(-40, 40), ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
        ax.set_yticks(np.arange(-40, 50, 10))
        ax.legend(frameon=False, loc='upper center')

        fname = "plot_test_reduction" + "_ci-" + ci_method
        suf = '_'  + model_plot
        fname += suf + '.png'

        plt.savefig(save_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.close()

    #endregion

    # .... 1.3. Plot number of tests ....
    #region

    ## Data for model
    data = dfgain.loc[(dfgain.model_name == model_plot) & (dfgain.metric_name=='pp_mod')]
    for col in ['metric_value', 'ci_low', 'ci_high']:
        data[col] = data[col] / data.n * 1000  # scale to 1000 tests
    data['add_low'] = data.ci_low - data.metric_value
    data['add_high'] = data.ci_high - data.metric_value
    data.period = data.period.replace(time_label_plot)
    ci = data[['add_low', 'add_high']].transpose().abs().to_numpy()

    x_model = np.arange(data.period.nunique())
    x_model = x_model - 0.15

    ## Data for model method 2
    data2 = dfgain2.loc[(dfgain2.model_name == model_plot) & (dfgain2.metric_name=='pp_mod')]
    for col in ['metric_value', 'ci_low', 'ci_high']:
        data2[col] = data2[col] / data2.n * 1000  # scale to 1000 tests
    data2['add_low'] = data2.ci_low - data2.metric_value
    data2['add_high'] = data2.ci_high - data2.metric_value
    data2.period = data2.period.replace(time_label_plot)
    ci2 = data2[['add_low', 'add_high']].transpose().abs().to_numpy()


    ## Data for FIT, method 1
    fit_data = dfgain.loc[(dfgain.model_name == model_plot) & (dfgain.metric_name=='pp_fit')]
    for col in ['metric_value', 'ci_low', 'ci_high']:
        fit_data[col] = fit_data[col] / fit_data.n * 1000  # scale to 1000 tests
    fit_data['add_low'] = fit_data.ci_low - fit_data.metric_value
    fit_data['add_high'] = fit_data.ci_high - fit_data.metric_value
    fit_data.period = fit_data.period.replace(time_label_plot)
    fit_ci = fit_data[['add_low', 'add_high']].transpose().abs().to_numpy()

    x_fit = np.arange(fit_data.period.nunique())
    x_fit = x_fit + 0.15

    ## Data for FIT, method 2
    fit_data2 = dfgain2.loc[(dfgain2.model_name == model_plot) & (dfgain2.metric_name=='pp_fit')]
    for col in ['metric_value', 'ci_low', 'ci_high']:
        fit_data2[col] = fit_data2[col] / fit_data2.n * 1000  # scale to 1000 tests
    fit_data2['add_low'] = fit_data2.ci_low - fit_data2.metric_value
    fit_data2['add_high'] = fit_data2.ci_high - fit_data2.metric_value
    fit_data2.period = fit_data2.period.replace(time_label_plot)
    fit_ci2 = fit_data2[['add_low', 'add_high']].transpose().abs().to_numpy()


    ## Plot
    line_plot = False
    fig, ax = plt.subplots(figsize=(12, 6))
    if not line_plot:
        ax.bar(x_model, data2.metric_value.to_numpy(), yerr=ci2, capsize=5, 
               color='C0', edgecolor=None, label=model_label,
               width=0.3)
        #ax.errorbar(x_model, data2.metric_value.to_numpy(), yerr=ci, capsize=0, 
        #            alpha = 0.7, marker='None', color='gray',linestyle='None')

        ax.bar(x_fit, fit_data2.metric_value.to_numpy(), yerr=fit_ci2, capsize=5, 
               color='C1', edgecolor=None, label="FIT ≥ 10 µg/g",
               width=0.3)
        #ax.errorbar(x_fit, fit_data2.metric_value.to_numpy(), yerr=fit_ci, capsize=0, 
        #            alpha = 0.7, marker='None', color='gray',linestyle='None')
    else:
        ax.plot(x_model, data2.metric_value.to_numpy(), color='C0', label=model_label, alpha=0.9)
        ax.scatter(x_model, data2.metric_value.to_numpy(), color='C0', s=64, alpha=0.9)
        ax.errorbar(x_model, data2.metric_value, yerr=ci2, alpha=0.9)
        ax.fill_between(x=x_model, y1=data2.ci_low, y2=data2.ci_high, alpha=0.15)

        ax.plot(x_model, data.ci_low.to_numpy(), color='C0', linestyle='dashed', alpha=0.75)
        ax.plot(x_model, data.ci_high.to_numpy(), color='C0', linestyle='dashed', alpha=0.75)

        ax.plot(x_fit, fit_data2.metric_value.to_numpy(), color='C1', label="FIT ≥ 10 µg/g", alpha=0.9)
        ax.scatter(x_fit, fit_data2.metric_value.to_numpy(), color='C1', s=64, alpha=0.9)
        ax.errorbar(x_fit, fit_data2.metric_value, yerr=fit_ci2, alpha=0.9)
        ax.fill_between(x=x_fit, y1=fit_data2.ci_low, y2=fit_data2.ci_high, alpha=0.15, color='C1')

    #ax.plot(x_fit, fit_data.ci_low.to_numpy(), color='C1', linestyle='dashed')
    #ax.plot(x_fit, fit_data.ci_high.to_numpy(), color='C1', linestyle='dashed')

    ax.grid(axis='y', linestyle='dotted')
    ax.legend(frameon=False, loc='upper center')
    ax.set(ylabel="Number of positive tests per 1000 tests")
    ax.set_xticks(np.arange(data.period.nunique()))
    ax.set_xticklabels(list(time_label_plot.values()))

    fname = "plot_num_test"
    suf = '_'  + model_plot
    fname += suf + '.png'

    plt.savefig(save_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()
    #endregion

    # .... 1.4. Plot percent reduction for all models ....
    #region
    models_plot = ['nottingham-lr', 'nottingham-cox', 'nottingham-lr-boot', 'nottingham-cox-boot']
    data = dfgain.loc[(dfgain.model_name.isin(models_plot)) & (dfgain.metric_name=='proportion_reduction_tests')]
    data['add_low'] = data.ci_low - data.metric_value
    data['add_high'] = data.ci_high - data.metric_value
    data.period = data.period.replace(time_label_plot)

    models_plot = ['nottingham-lr', 'nottingham-cox', 'nottingham-lr-boot', 'nottingham-cox-boot']
    data2 = dfgain2.loc[(dfgain2.model_name.isin(models_plot)) & (dfgain2.metric_name=='proportion_reduction_tests')]
    data2['add_low'] = data2.ci_low - data2.metric_value
    data2['add_high'] = data2.ci_high - data2.metric_value
    data2.period = data2.period.replace(time_label_plot)

    # Get unique periods and categories
    periods_plot = data['period'].unique()

    # Define bar width and positions (from ChatGPT)

    x = np.arange(len(periods))

    bar = False
    method1 = True

    if bar:
        bar_width = 0.2
    else:
        bar_width = 0.15

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models_plot):
        cat_data = data2[data2.model_name == model]

        cat_data1 = data[data.model_name == model]
        #ax.scatter(x + i * bar_width,
        #           cat_data['metric_value'], 
        #           s=64)
        if not bar:
            ax.errorbar(x + i * bar_width, 
                        cat_data['metric_value'],
                        yerr=[cat_data['metric_value'] - cat_data['ci_low'], cat_data['ci_high'] - cat_data['metric_value']],
                        alpha = 0.9, fmt='.', markersize=18,
                        label=model_labels[model], color='C' + str(i))
            
            if method1:
                ax.errorbar(x + i * bar_width, 
                            cat_data1['metric_value'],
                            yerr=[cat_data1['metric_value'] - cat_data1['ci_low'], cat_data1['ci_high'] - cat_data1['metric_value']],
                            alpha = 0.33, marker='None', color='gray', zorder=-1,
                            linestyle='None')
            
            #if i == 0:
            #   ax.fill_betweenx(y=[-30, 30], x1 = x, x2 = x + 4 * bar_width, facecolor='gray', alpha=0.25)
        else:
            ax.bar(x + i * bar_width,
                cat_data['metric_value'],
                width=bar_width,
                yerr=[cat_data['metric_value'] - cat_data['ci_low'], cat_data['ci_high'] - cat_data['metric_value']],
                capsize=5,
                label=model_labels[model])

    ax.hlines(y=0, xmin=min(x) - 0.2, xmax = max(x) + 0.6 + 0.2, linestyle='dashed', color='red')
    ax.grid(axis='y', linestyle='dotted')
    ax.set_xticks(x + bar_width * (len(models_plot) - 1) / 2)
    ax.set_xticklabels(periods_plot)
    #ax.set(ylim=(-50, 100), ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
    #ax.set_yticks(np.arange(-50, 110, 10))
    ax.set(ylim=(-40, 40), ylabel="Percent reduction in the number of positive tests\ncompared to FIT ≥ 10 µg/g (negative is better)")
    ax.set_yticks(np.arange(-40, 50, 10))
    ax.legend(frameon=False)

    fname = "plot_test_reduction" + "_ci-" + ci_method
    suf = '_allmod'
    fname += suf + '.png'

    plt.savefig(save_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()
    #endregion
    
    # ---- 2. Plot discrimination, calibration and decision curves ----

    # Time period labels and colors
    #region
    if fu == 365:
        time_labels = {
                    'all': 'All data (2017/01 - 2023/02)',
                    'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
                    'bufftime': 'Buffer data from time (2021/10 - 2023/02)',
                    'buffcomm': 'Buffer data from comment (2022/04 - 2023/02)',
                    }
    else:
        time_labels = {
                    'all': 'All data (2017/01 - 2023/08)',
                    'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
                    'bufftime': 'Buffer data from time (2021/10 - 2023/08)',
                    'buffcomm': 'Buffer data from comment (2022/04 - 2023/08)',
                    }

    time_colors = {
                'all': 'C0',
                'prebuff': 'C1',
                'bufftime': 'C2',
                'buffcomm': 'C3'
                }
    #endregion

    # .... 2.1. ROC curves ....
    #region
    for ci_roc in [True, False]:
        __, __ = plot_rocpr_agg(save_path, model_plot, periods, time_labels, time_colors, ci=ci_roc, xlow=None)
        __, __ = plot_rocpr_agg(save_path, model_plot, periods, time_labels, time_colors, ci=ci_roc, xlow=0.7)
        __, __ = plot_rocpr_agg(save_path, model_plot, periods, time_labels, time_colors, ci=ci_roc, xlow=None,
                                models_plot2=['nottingham-cox', 'fit'], suf2="_cox")
        __, __ = plot_rocpr_agg(save_path, model_plot, periods, time_labels, time_colors, ci=ci_roc, xlow=0.7,
                                models_plot2=['nottingham-cox', 'fit'], suf2="_cox")

    for ci_roc in [True, False]:
        plot_cal_smooth_agg(save_path, model_plot, periods, time_labels, time_colors, ci=ci_roc)
    #endregion

    # .... 2.2. Calibration curves ....
    #region
    models_add = ['nottingham-lr-platt', 'nottingham-lr-3.5', 'nottingham-lr-quant']
    models_add2 = ['nottingham-cox-platt', 'nottingham-cox-3.5', 'nottingham-cox-quant']

    for ci_plot in [True, False]:
        plot_cal_smooth_agg(save_path, model_plot=None, periods=periods, time_labels=time_labels, time_colors=time_colors, ci=ci_plot,
                            models_incl=models_add, suf2='_recal-lr')
        
        plot_cal_smooth_agg(save_path, model_plot=None, periods=periods, time_labels=time_labels, time_colors=time_colors, ci=ci_plot,
                            models_incl=models_add2, suf2='_recal')
    #endregion

    # .... 2.3. Decision curves ....
    #region
    if 'quant' in model_plot:
        model_plot_dca = model_plot
    else:
        model_plot_dca = model_plot + '-quant'
    model_colors2 = {key: val for key, val in model_colors.items()}
    model_colors2['nottingham-cox-quant'] = 'C0'
    model_colors2['fit-spline'] = 'C1'
    for ci_plot in [True, False]:
        plot_dca_agg(save_path, periods=periods, time_labels=time_labels, model_labels=model_labels,
                     model_colors=model_colors2, ni = False, ci = ci_plot, models_incl = [model_plot_dca, 'fit-spline'],
                     suf='_' + model_plot_dca)
    #endregion