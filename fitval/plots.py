"""Visualise discrimination and calibration"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fitval.models import model_labels, model_colors
from pathlib import Path
from fitval.boot import CAL_SMOOTH_CI, CAL_BIN_NOCI, ROC_NOCI, PR_NOCI, ROC_CI, PR_CI, PR_GAIN_CI, SENS_FIT_CI, DC_CI
from matplotlib.ticker import MaxNLocator
import string


# Output files
PLOT_CAL_BIN = 'plot_calibration_binned'
PLOT_CAL_SMOOTH = 'plot_calibration_smooth'
ROCPR_PLOT = 'plot_rocpr'
ROCPR_INTERP_PLOT = 'plot_rocpr_interp'  # .png and attributes added later depending on plot settings
PLOT_DCA = 'plot_dca'  # .png and attributes added later depending on plot settings


# ~ Calibration curves ~
def plot_cal_bin(run_path: Path, model_labels: dict = model_labels,
                 model_colors: dict = model_colors, figsize: tuple = (6, 5), 
                 boot_alpha: float = 0.1, exclude_boot: bool = False, grid: bool = False,
                 models_incl: list = None, suf: str = ''):
    """Plot binned calibration curves
    Main args
        df: DataFrame with columns:
            model_name: model_name
            b: bootstrap sample
            m: imputation (M imputations are conducted for each bootstrap sample)
            strategy: strategy of creating calibration curve, 'quantile' or 'uniform'
            fit_ub: upper bound that was applied to FIT values when calibration curve was created
        run_path: Path where plots are to be saved
    
    A single plot is created for each strategy and fit_ub combination
    """
    print('\nPlotting binned calibration curves...')

    df = pd.read_csv(run_path / CAL_BIN_NOCI)
    if 'metric_name' in df.columns:
        name = 'prob_true'
        df = df.loc[df.metric_name == name].rename(columns={'metric_value': name})

    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()  # Models to be plotted

    # Do not plot bootstrap samples
    if exclude_boot:
        df = df.loc[df['b'] == -1]

    # Groups: one plot per group
    df.loc[df.ymax > 0.2, 'ymax'] = 1  # For file names and grouping only
    df.loc[df.ymax < 0.2, 'ymax'] = 0.2  # For file names and grouping only
    groups = df[['strategy', 'fit_ub', 'ymax']].drop_duplicates()

    # Models, samples, imputations: shown on a single plot within group
    samples = df['b'].unique()  # Samples for each model to be plotted (-1: original sample, 0: first bootstrap sample etc)
    imputations = df['m'].unique()

    # Create one plot per group
    for __, row in groups.iterrows():

        # Data to be plotted: for current strategy (strategy) and FIT upper bound (fit_ub)
        group_mask = (df.strategy == row.strategy) & (df.fit_ub == row.fit_ub) & (df.ymax == row.ymax)
        dfg = df.loc[group_mask]
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

        for model_name in models:
            for b in samples:
                for m in imputations:

                    # Data to be plotted: for current model (model_name), bootstrap sample (b), and imputation (m)
                    dfsub = dfg.loc[(dfg.model_name == model_name) & (dfg.b == b) & (dfg.m == m)]

                    # Adjust plotting settings depending on type of sample (-1: original, >-1: bootstrap)
                    if b == -1:
                        alpha = 0.75
                        if m == 0:
                            label = model_labels[model_name]
                        else:
                            label = None
                        zorder = 3
                        linewidth = 1.5
                    else:
                        alpha = boot_alpha
                        label = None
                        zorder = 2
                        linewidth = 0.5
                    
                    # Get data for current sample
                    prob_pred = dfsub.prob_pred.to_numpy()
                    prob_true = dfsub.prob_true.to_numpy()

                    # Sort
                    idx = np.argsort(prob_pred)
                    prob_pred = prob_pred[idx]
                    prob_true = prob_true[idx]

                    # Plot line
                    ax.plot(prob_pred, prob_true, color=model_colors[model_name], alpha=alpha, 
                            linestyle='solid', zorder=zorder, label=label, linewidth=linewidth)
                    
                    # Plot ideal line, and format
                    if (b == -1) & (m == 0):
                        ax.scatter(prob_pred, prob_true, color=model_colors[model_name], alpha=alpha, 
                                linestyle='solid', zorder=zorder, label=None, s=5)

                        # Ideal calibration line
                        eps = 0.
                        ax.plot([0-eps, 1+eps], [0-eps, 1+eps], color='gray', alpha=1, linewidth=1, linestyle='dashed', zorder=1)

                        # Adjustments
                        ymax = row.ymax
                        step = ymax / 10
                        xticks = np.arange(0, ymax + step, step)
                        if ymax in [0.2, 1]:
                            xlabels = (xticks * 100).astype(int)
                        else:
                            xlabels = np.round(xticks * 100, 1)
                        
                        eps = step / 2
                        xlim = (0 - eps, ymax + eps)
                        if ymax <= 0.5:
                            yticks = np.arange(0, 0.5 + 0.05, 0.05)
                            ylim=(0 - 0.025, 0.5 + 0.025)
                        else:
                            yticks = np.arange(0, 1.1, 0.1)
                            ylim=(0 - eps, 1 + eps)
                        ylabels = (yticks * 100).astype(int)

                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xlabels)
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(ylabels)
                        ax.set(xlabel='Predicted probability (%)', ylabel='Observed probability (%)', title=None,
                            xlim=xlim, ylim=ylim)
                        ax.legend(frameon=False)

                        if grid:
                            ax.grid(which='major', zorder=1, alpha=0.5)
                            #ax.minorticks_on()
                            #ax.grid(which='minor')

        out_name = PLOT_CAL_BIN + '_ymax-' + str(row.ymax) + '_strategy-' + row.strategy + \
                   '_fitupbound-' + str(row.fit_ub) + suf + '.png'
        plt.savefig(run_path / out_name, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.close()


def plot_cal_smooth(run_path: Path, model_labels: dict = model_labels, 
                    model_colors: dict = model_colors, figsize: tuple = (6, 5), grid: bool = False,
                    models_incl: list = None, suf: str = ''):
    """Plot smooth calibration curves"""
    print('\nPlotting smooth calibration curves...')

    df = pd.read_csv(run_path / CAL_SMOOTH_CI)
    if 'metric_name' in df.columns:
        name = 'prob_true'
        df = df.loc[df.metric_name == name].rename(columns={'metric_value': name})

    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()

    # Spline curves don't have frac, and lowess curves don't have knots 
    # just in case fill in missing values to identify groups correctly
    df.frac = df.frac.fillna('none')
    #df.n_knots = df.n_knots.fillna('none')
    #df.strategy = df.strategy.fillna('none')
    df.fit_ub = df.fit_ub.fillna('none')
    df.loc[df.ymax > 0.2, 'ymax'] = 1  # For file names only
    df.loc[df.ymax < 0.2, 'ymax'] = 0.2   # For file names only

    groups = df[['frac', 'fit_ub', 'ymax']].drop_duplicates()

    for __, row in groups.iterrows():
        #mask = (df.strategy == row.strategy) & (df.n_knots == row.n_knots)
        mask = (df.frac == row.frac) & (df.fit_ub == row.fit_ub) & (df.ymax == row.ymax)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
        
        for m in models:
            dfsub = df.loc[(df.model_name == m) & mask]
            #dfsub = dfsub.loc[dfsub.prob_true < 1]
            
            # Calibration curve with confidence intervals
            ax.plot(dfsub.prob_pred, dfsub.prob_true, color=model_colors[m], alpha=0.75, 
                    linestyle='solid', label=model_labels[m], zorder=2)
            ax.fill_between(dfsub.prob_pred, dfsub.ci_low, dfsub.ci_high, alpha=0.25, 
                            facecolor=model_colors[m], edgecolor=None)

            # Ideal calibration line
            eps = 0.
            ax.plot([0-eps, 1+eps], [0-eps, 1+eps], color='gray', alpha=1, linewidth=1, linestyle='dashed', zorder=1)

            # Adjustments
            ymax = row.ymax
            step = ymax / 10
            xticks = np.arange(0, ymax + step, step)
            if ymax in [0.2, 1]:
                xlabels = (xticks * 100).astype(int)
            else:
                xlabels = np.round(xticks * 100, 1)
            
            eps = step / 2
            xlim = (0 - eps, ymax + eps)
            if ymax <= 0.5:
                yticks = np.arange(0, 0.5 + 0.05, 0.05)
                ylim=(0 - 0.025, 0.5 + 0.025)
            else:
                yticks = np.arange(0, 1.1, 0.1)
                ylim=(0 - eps, 1 + eps)
            ylabels = (yticks * 100).astype(int)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
            ax.set(xlabel='Predicted probability (%)', ylabel='Observed probability (%)', title=None,
                xlim=xlim, ylim=ylim)
            ax.legend(frameon=False)

            if grid:
                ax.grid(which='major', zorder=1, alpha=0.5)
                #ax.minorticks_on()
                #ax.grid(which='minor')

        # '_strategy-' + row.strategy + '_nknots-' + str(row.n_knots) +
        out_name = PLOT_CAL_SMOOTH + '_ymax-' + str(row.ymax) + '_frac-' + str(row.frac) + \
                   '_fitupbound-' + str(row.fit_ub) + suf + '.png'
        plt.savefig(run_path / out_name, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.close()


# ~ Discrimination curves ~
def plot_rocpr(run_path: Path, model_labels: dict = model_labels,
               model_colors: dict = model_colors, figsize: tuple = (12, 6), hspace: float = 0.3,
               exclude_boot: bool = True, boot_alpha: float = 0.1, boot_color: str = None, grid: bool = False,
               models_incl: list = None, suf: str = ''):
    """Plot ROC and precision-recall curves
        roc: DataFrame for roc curve, contains columns 'tpr', 'fpr' for each model
        pr: DataFrame for pr curve, contains columns 'recall', 'precision' for each model
        run_path: where to save results
    """
    print('\nPlotting ROC and PR curves...')
    fname = ROCPR_PLOT + suf + '.png'

    # Load non-interpolated data
    roc = pd.read_csv(run_path / ROC_NOCI)
    pr = pd.read_csv(run_path / PR_NOCI)

    if models_incl is not None:
        models = models_incl
        #roc = roc.loc[roc.model_name.isin(models_incl)]
        #pr = pr.loc[pr.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in pr.model_name.unique()]
    else:
        models = roc.model_name.unique()

    # Do not display curves from bootstrap samples?
    if exclude_boot:
        roc = roc.loc[roc['b'] == -1]
        pr = pr.loc[pr['b'] == -1]

    # Plot
    samples = roc['b'].unique()
    imputations = roc['m'].unique()

    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)
    ax = ax.flatten()

    for m in models:
        for s in samples:
            for i in imputations:
                if (m == 'fit') and (i > 0):  # FIT test is not imputed, its i is set to 0
                    continue

                if s == -1:
                    alpha = 0.75
                    if i == 0:
                        label = model_labels[m]
                    else:
                        label = None
                    zorder = 2
                    color = model_colors[m]
                else:
                    alpha = boot_alpha
                    label = None
                    zorder = 3
                    color = model_colors[m] if boot_color is None else 'gray'

                u = roc.loc[(roc.model_name == m) & (roc['b'] == s) & (roc['m'] == i)]
                v = pr.loc[(pr.model_name == m) & (pr['b'] == s) & (pr['m'] == i)]

                ax[0].plot(u.fpr * 100, u.tpr * 100, label=label, alpha=alpha, color=color, zorder=zorder)
                ax[1].plot(v.recall * 100, v.precision * 100, label=label, alpha=alpha, color=color, zorder=zorder)

                if (s == -1) & (i == 0):
                    ax[0].set(xlabel='False positive rate',
                              ylabel='Sensitivity (% cancers detected)',
                              title='ROC curve')
                    ax[0].legend(frameon=False)
                    ax[0].set_xticks(np.arange(0, 1.1, 0.1) * 100)
                    ax[0].set_yticks(np.arange(0, 1.1, 0.1) * 100)

                    ax[1].set(xlabel='Sensitivity (% cancers detected)',    
                              ylabel='Positive predictive value',
                              title='Precision-recall curve')#,
                              #ylim=(0, None))
                    ax[1].legend(frameon=False)
                    ax[1].set_xticks(np.arange(0, 1.1, 0.1) * 100)
                    ax[1].set_yticks(np.arange(0, 1.1, 0.1) * 100)

                    if grid:
                        for z in [0, 1]:
                            ax[z].grid(which='major', zorder=1, alpha=0.5)
                            #ax[z].minorticks_on()
                            #ax[z].grid(which='minor')

    plt.subplots_adjust(hspace=hspace)
    plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')


def plot_rocpr_interp(run_path: Path, model_labels: dict = model_labels, model_colors: dict = model_colors,
                      figsize: tuple = (18, 6), hspace: float = 0.5, grid: bool = False,
                      models_incl: list = None, ci: bool = True, suf: str = '',
                      show_reduction_in_tests: bool = True, sens_fit10: bool = True):
    """Plot ROC, precision-recall, and relative precision-gain curves for each model-loss combination.
    Using interpolated data, so that data of all folds can be plotted on a single plot with shading.

        roc: DataFrame for INTERPOLATED roc curve, contains columns 'tpr', 'fpr' for each model-loss-fold
        pr: DataFrame for INTERPOLATED pr curve, contains columns 'recall', 'precision' for each model-loss-fold
        run_path: where to save results
        shade: 'minmax' - boundaries of shaded area show min and max values over cross-validation folds
               'sd' - boundaries of shaded area show mean +/- standard deviation of values over cross-validation folds
    
    Note: if the first point on precision-recall curve (for recall > 0) is rather far from 0, 
    the data for lower recalls are filled backwards.
    """
    print('\nPlotting interpolated ROC and PR curves...')
    fname = ROCPR_INTERP_PLOT
    if ci:
        fname += '_ci-95'
    else:
        fname += '_ci-none'
    fname += suf + '.png'

    if sens_fit10:
        s = pd.read_csv(run_path / SENS_FIT_CI)
        s = s.loc[(s.model == 'fit') & (s.thr_fit == 10) & (s.metric_name == 'sens')]
        sens_fit = s.metric_value.item() * 100

    # Load interpolated data
    roc = pd.read_csv(run_path / ROC_CI)
    pr = pd.read_csv(run_path / PR_CI)

    if 'metric_name' in roc.columns:
        roc = roc.loc[roc.metric_name == 'tpr'].rename(columns={'metric_value': 'tpr'})

    if 'metric_name' in pr.columns:
        sens_max = pr.loc[pr.metric_name=='max_recall'].rename(columns={'metric_value': 'max_recall'})
        sens_max = sens_max.loc[sens_max.model_name=='fit'].max_recall.max()
        pr = pr.loc[pr.metric_name == 'precision'].rename(columns={'metric_value': 'precision'})
    else:
        sens_max = pr.loc[pr.model_name=='fit'].max_recall.max()

    if models_incl is not None:
        models = models_incl
        #roc = roc.loc[roc.model_name.isin(models_incl)]
        #pr = pr.loc[pr.model_name.isin(models_incl)] 
        models = [mod for mod in models if mod in pr.model_name.unique()] 
    else:
        models = roc.model_name.unique()

    pr_gain = pd.read_csv(run_path / PR_GAIN_CI)
    if 'metric_name' in pr_gain.columns:
        name = 'proportion_reduction_tests'
        test_red = pr_gain.loc[pr_gain.metric_name == name].rename(columns={'metric_value': name})

        name = 'precision_gain'
        pr_gain = pr_gain.loc[pr_gain.metric_name == name].rename(columns={'metric_value': name})

    #test_red = pd.read_csv(run_path / TEST_REDUCTION_CI)

    # Load non-interpolated data to get max sens of FIT 
    #pr_noint = pd.read_csv(run_path / PR_NOCI)
    #pr_noint = pr_noint.loc[pr_noint.model_name=='fit']
    test_red = test_red.loc[test_red.recall <= sens_max]
    pr_gain = pr_gain.loc[pr_gain.recall <= sens_max]

    # Rescale data to percentage
    roc, pr, pr_gain, test_red = roc.copy(), pr.copy(), pr_gain.copy(), test_red.copy()
    scale = 100
    roc[['tpr', 'fpr', 'ci_low', 'ci_high']] *= scale
    pr[['recall', 'precision', 'ci_low', 'ci_high']] *= scale
    pr_gain[['recall', 'precision_gain', 'ci_low', 'ci_high']] *= scale
    test_red[['recall', 'proportion_reduction_tests', 'ci_low', 'ci_high']] *= scale

    fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

    for m in models:
        u = roc.loc[(roc.model_name == m)]
        v = pr.loc[(pr.model_name == m)]
        w = pr_gain.loc[(pr_gain.model_name == m)]
        z = test_red.loc[(test_red.model_name == m)]

        # ROC curve
        ax[0].plot(u.fpr, u.tpr, label=model_labels[m], alpha=0.75, color=model_colors[m])
        #ax[0].fill_betweenx(u.tpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
        if ci:
            ax[0].fill_between(u.fpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
        if sens_fit10:
            ax[0].axhline(y=sens_fit, color='gray', linestyle='solid', alpha=0.3)

        # PR curve
        ax[1].plot(v.recall, v.precision, label=model_labels[m], alpha=0.75, color=model_colors[m])
        if ci:
            ax[1].fill_between(v.recall, v.ci_low, v.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
        if sens_fit10:
            ax[1].axvline(x=sens_fit, color='gray', linestyle='solid', alpha=0.3)

        # Gain in PR curve
        if m != 'fit':
            if show_reduction_in_tests:
                ax[2].plot(z.recall, z.proportion_reduction_tests, label=model_labels[m], alpha=0.75, color=model_colors[m])
                if ci:
                    ax[2].fill_between(z.recall, z.ci_low, z.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
                if sens_fit10:
                    ax[2].axvline(x=sens_fit, color='gray', linestyle='solid', alpha=0.3)
    
            else:
                ax[2].plot(w.recall, w.precision_gain, label=model_labels[m], alpha=0.75, color=model_colors[m])
                if sens_fit10:
                    ax[2].axvline(x=sens_fit, color='gray', linestyle='solid', alpha=0.3)
                if ci:
                    ax[2].fill_between(w.recall, w.ci_low, w.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
    
    # Adjust
    ax[0].set(xlabel='False positive rate', ylabel='Sensitivity (% cancers detected)', title='ROC curve')
    ax[0].legend(frameon=False, loc='best')
    ax[0].set_xticks(np.arange(0, 1.1, 0.1) * scale)
    ax[0].set_yticks(np.arange(0, 1.1, 0.1) * scale)

    ax[1].set(xlabel='Sensitivity (% cancers detected)', ylabel='Positive predicive value',
              title='Precision-recall curve',
              ylim=(0, None))
    ax[1].legend(frameon=False, loc='best')
    ax[1].set_xticks(np.arange(0, 1.1, 0.1) * scale)
    ax[1].set_yticks(np.arange(0, 1.1, 0.1) * scale)

    if show_reduction_in_tests:
        ax[2].hlines(y=0, xmin=0, xmax=1*scale, color='red', linestyles='dashed', alpha=0.5, label=model_labels['fit'])
        ax[2].set(xlabel='Sensitivity (% cancers detected)', ylabel='Percent reduction in number of tests (lower is better)',
                  title='Test reduction curve')
        #loc = 'lower right' if pr_gain.precision_gain.min() < 0 else 'best'
        ax[2].legend(frameon=False, loc='best')
        ax[2].set_xticks(np.arange(0, 1.1, 0.1) * scale)
    else:
        ax[2].hlines(y=0, xmin=0, xmax=1*scale, color='red', linestyles='dashed', alpha=0.5, label=model_labels['fit'])
        ax[2].set(xlabel='Sensitivity (% cancers detected)', ylabel='Gain in positive predictive value', title='PPV gain curve')
        #loc = 'lower right' if pr_gain.precision_gain.min() < 0 else 'best'
        ax[2].legend(frameon=False, loc='best')
        ax[2].set_xticks(np.arange(0, 1.1, 0.1) * scale)

    if grid:
        for z in [0, 1, 2]:
            ax[z].grid(which='major', zorder=1, alpha=0.5)
            #ax[z].minorticks_on()
            #ax[z].grid(which='minor')

    plt.subplots_adjust(hspace=hspace)
    plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()


# ~ Net benefit curves ~
def plot_dca(run_path: Path, model_labels: dict = model_labels, model_colors: dict = model_colors, 
             figsize: tuple = (6, 5), xlim: tuple = (None, None), grid: bool = False, ni: bool = False,
             models_incl: list = None, ci: bool = True, suf: str = ''):
    print('\nPlotting decision analysis curves...')

    # Output file name
    fname = PLOT_DCA
    if ni:
        fname += '_avoided'
    else:
        fname += '_benefit'
    if ci:
        fname += '_ci-95'
    else:
        fname += '_ci-none'
    if xlim[0] or xlim[1]:
        fname += '_xlim-' + str(xlim)
    fname += suf
    fname += '.png'

    # Data, y-axis value column, y-axis label
    df = pd.read_csv(run_path / DC_CI)
    if ni:
        col = 'net_intervention_avoided'
        lab = 'Net intervention avoided'
    else:
        col = 'net_benefit'
        lab = 'Net benefit'
    
    df = df.loc[df.metric_name == col].rename(columns={'metric_value': col})
    df = df.sort_values(by=['model_name', 'threshold', col], ascending=[True, True, False])
    
    # Exclude any models?
    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()

    # x axis limit?
    if xlim[0] is not None:
        df = df.loc[df.threshold >= xlim[0] / 100]
    if xlim[1] is not None:
        df = df.loc[df.threshold <= xlim[1] / 100]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    eps = df[col].max() - 0.
    ymax = df[col].max() + eps/10
    ymin = 0. - eps/10

    # Note: the 'all', 'none', and 'fit10' were computed separately under each model,
    # but are the same for all models
    all = df.loc[(df.model == 'all') & (df.model_name == models[0])]
    none = df.loc[(df.model == 'none') & (df.model_name == models[0])]
    fit10 = df.loc[(df.model == 'fit10') & (df.model_name == models[0])]
    model = df.loc[df.model == 'model']

    # Plot net benefit curves for models
    for model_name in models:
        mod = model.loc[df.model_name == model_name]
        ax.plot(mod.threshold * 100, mod[col], label=model_labels[model_name], color=model_colors[model_name], 
                linestyle='solid', linewidth=1, alpha=0.75)
        if ci:
            ax.fill_between(mod.threshold * 100, mod.ci_low, mod.ci_high, color=model_colors[model_name], alpha=0.2, edgecolor=None)
    
    # Plot net benefit line for FIT >= 10
    ax.plot(fit10.threshold * 100, fit10[col], label='FIT >= 10', color='red', linestyle='dashed', linewidth=1, alpha=0.75)
    if ci:
        ax.fill_between(fit10.threshold * 100, fit10.ci_low, fit10.ci_high, color='red', alpha=0.2, edgecolor=None)

    # Plot net benefit lines for 'treat all' and 'treat none'
    ax.plot(all.threshold * 100, all[col], label='Treat All', color='gray', linestyle='solid', linewidth=1, alpha=0.75)
    ax.plot(none.threshold * 100, none[col], label='Treat None', color='gray', linestyle='dashed', linewidth=1, alpha=0.75)

    ax.set(ylim=(ymin, ymax), xlabel="Predicted risk (%)", ylabel=lab)
    thr = df.threshold
    step = thr.max() * 0.1
    xticks = np.arange(start=0, stop=thr.max() + step, step=step) * 100
    #xticks = xticks.astype(int)
    ax.set_xticks(xticks)

    ax.legend(frameon=False, bbox_to_anchor=(1.0, 1))

    if grid:
        ax.grid(which='major', zorder=1, alpha=0.5)
        #ax[z].minorticks_on()
        #ax[z].grid(which='minor')

    plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()


def plot_dca_panel(run_path: Path, model_labels: dict = model_labels, model_colors: dict = model_colors, 
                   figsize: tuple = (18, 6), ni: bool = False,
                   models_incl: list = None, ci: bool = False, suf: str = ''):
    print('\nPlotting decision analysis curves...')

    # Output file name
    fname = PLOT_DCA

    if ni:
        fname += '_avoided'
    else:
        fname += '_benefit'

    if ci:
        fname += '_ci-95'
    else:
        fname += '_ci-none'

    fname += suf
    fname += '_panel.png'

    # Data, y-axis value column, y-axis label
    df = pd.read_csv(run_path / DC_CI)
    if ni:
        col = 'net_intervention_avoided'
        lab = 'Net intervention avoided'
    else:
        col = 'net_benefit'
        lab = 'Net benefit'
    
    df = df.loc[df.metric_name == col].rename(columns={'metric_value': col})
    df = df.sort_values(by=['model_name', col])

    
    # Exclude any models?
    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    cut = [5, 20, 50]
    for i, c in enumerate(cut):
        dfsub = df.loc[df.threshold <= c / 100]
        title = 'Risk ≤ ' + str(c) + ' %'
        _plot_dca(dfsub, col, lab, models, ax[i], ci, model_colors, model_labels, title)

    if ni:
        for i in [0, 1, 2]:
            ax[i].legend(frameon=False, loc='lower right')
    else:
        for i in [1, 2]:
            ax[i].legend(frameon=False, loc='upper right')
        ax[0].legend(frameon=False, loc='lower left')
    plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()


def _plot_dca(df, col, lab, models, ax, ci, model_colors, model_labels, title):
    print('\nPlotting decision analysis curves...')

    eps = df[col].max() - 0.
    ymax = df[col].max() + eps/10
    ymin = 0. - eps/10

    # Note: the 'all', 'none', and 'fit10' were computed separately under each model,
    # but are the same for all models
    all = df.loc[(df.model == 'all')]
    none = df.loc[(df.model == 'none')]
    fit10 = df.loc[(df.model == 'fit10')]
    model = df.loc[df.model == 'model']

    # Plot net benefit curves for models
    for model_name in models:
        mod = model.loc[df.model_name == model_name]
        mod = mod.sort_values(by=['threshold', 'net_benefit'], ascending=[True, False])
        ax.plot(mod.threshold * 100, mod[col], label=model_labels[model_name], color=model_colors[model_name], 
                linestyle='solid', linewidth=1, alpha=0.75)
        if ci:
            ax.fill_between(mod.threshold * 100, mod.ci_low, mod.ci_high, color=model_colors[model_name], alpha=0.2, edgecolor=None)
    
    # Plot net benefit line for FIT >= 10
    ax.plot(fit10.threshold * 100, fit10[col], label='FIT >= 10', color='red', linestyle='dashed', linewidth=1, alpha=0.75)
    if ci:
        ax.fill_between(fit10.threshold * 100, fit10.ci_low, fit10.ci_high, color='red', alpha=0.2, edgecolor=None)

    # Plot net benefit lines for 'treat all' and 'treat none'
    ax.plot(all.threshold * 100, all[col], label='Treat All', color='gray', linestyle='solid', linewidth=1, alpha=0.75)
    ax.plot(none.threshold * 100, none[col], label='Treat None', color='gray', linestyle='dashed', linewidth=1, alpha=0.75)

    ax.set(ylim=(ymin, ymax), xlabel="Predicted risk (%)", ylabel=lab, title=title)
    thr = df.threshold
    step = thr.max() * 0.1
    xticks = np.arange(start=0, stop=thr.max() + step, step=step) * 100
    #xticks = xticks.astype(int)
    ax.set_xticks(xticks)
    ax.grid(which='major', zorder=1, alpha=0.5)


# ~ For plotting event logs ~
def plot_timeline(df, save_path, timecol='days_diagnosis_to_start', n=300, xmin=None, xmax=None, 
                  plotname='timeline', save=True, save_pdf=False, dpi=200, subj_order=None, 
                  all_ticks=False, subj_axis=False, hide_x=False, hide_y=False, vlines=True, anchor=1.25,
                  seed=1, title=None, groupvar=None, groupvarname=None,
                  subplots_vertical=False, title_loc='center',
                  nrow=None, ncol=None, w=None, h=None, pad=1, legend_idx=None, pad_inches=0.5,
                  gray=False, hide_axis=None, period=False, periodcol0='periodcol0', periodcol1='periodcol1',
                  perioddelta=1, vdelta=2, colors=None, event_order=None, xlabel=None, drop_diagnosis=True):
    
    # Map events to color and shape
    if xlabel is None:
        xlabel = 'Days since first known diagnosis'
    if colors is None:
        if gray:
            colors = { 'polypectomy': ['gray','o'],
               'colonic stent':     ['gray', 'o'],
               'colonoscopy' :      ['gray', 'o'],
               'radical resection': ['gray','o'], 
               'local excision':    ['gray','o'],
               'chemotherapy':      ['gray','o'],
               'radiotherapy':      ['gray','o'],
               'death':             ['gray','o'],
               'last alive':        ['gray','o'],
               'scan':              ['gray', 'o'],
               'TNM staging':       ['gray', 'o'],
               'suspicious for recurrence': ['gray', 'o'],
               'recurrence': ['gray', 'o'],
               'suspicious for metastasis': ['gray', 'o'],
               'metastasis': ['gray', 'o'],
              }
        else:
            colors = { 'polypectomy':       ['yellow','o'],
                       'colonic stent':     ['red', 2],
                       'colonoscopy' :      ['black', 2],
                       'radical resection': ['red','o'], 
                       'local excision':    ['orange','o'],
                       'chemotherapy':      ['blue','o'],
                       'radiotherapy':      ['green','o'],
                       'death':             ['gray','x'],
                       'last alive':        ['gray','o'],
                       'scan':              ['gray', 2],
                       'TNM staging':       ['violet', '.'],
                       'suspicious for recurrence': ['orange', 11],
                       'recurrence': ['red', 11],
                       'suspicious for metastasis': ['orange', 3],
                       'metastasis': ['red', 3],
                      }

    # Specify event order on legend
    if event_order is None:
        event_order = ['last alive', 'death', 'scan', 'colonoscopy',  'colonic stent', 'TNM staging',
                       'chemotherapy', 'radiotherapy', 
                       'polypectomy', 'local excision', 'radical resection',
                       'suspicious for recurrence', 'recurrence', 'suspicious for metastasis', 'metastasis']
    event_order = np.array(event_order)
    
    # Process the grouping variable and adjust plot height
    if groupvar is not None:  # If grouping variable is defined
        # Get values of the grouping variable and number of groups
        values = df[groupvar].unique()
        values = values[~pd.isnull(values)]
        if np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.int)
        values  = np.sort(values)
        #print(values)
        n_group = len(values)
        
        # Adjust plot height
        test = df.groupby(groupvar)['subject'].nunique().max()
        if n > test:
            n_vline = test
        else:
            n_vline = n
    else:  # If grouping variable is not defined
        # Set number of groups to 1, and values to a dummy list
        n_group = 1
        values = [1]
        
        # Adjust plot height
        test = df.subject.nunique()
        if n > test:
            n_vline = test
        else:
            n_vline = n
    
    # Drop diagnosis and birth as events
    if np.isin('diagnosis', df.event) and drop_diagnosis:
        df = df.loc[df.event != 'diagnosis']
    if np.isin('birth', df.event):
        df = df.loc[df.event != 'birth']

    # Draw canvas
    if n_vline < 15:
        height = 15/3.5
    else:
        height = n_vline/3.5
    
    if nrow is not None and ncol is not None:
        fig, ax = plt.subplots(nrow, ncol, figsize=(w, h))
        fig.tight_layout(pad=pad)
    else:
        fig, ax = plt.subplots(1, n_group, figsize=(12*n_group, height))
    
    if n_group > 1:
        ax = ax.flatten()
    else:
        ax = [ax]
    
    # Loop over values
    handles = np.empty(1)
    labels  = np.empty(1)
    for j, val in enumerate(values):
        
        # Data
        if n_group > 1:
            df_val = df.loc[df[groupvar] == val]
        else:
            df_val = df
        
        # Title
        if groupvarname is not None:
            title_group = groupvarname + '' + str(val)
        elif title is not None:
            title_group = title[j]
        else:
            title_group = title
        
        # Select n individuals randomly if requested
        subj = df_val.subject.unique() 
        if n < len(subj):
            rng     = np.random.default_rng(seed=seed)
            subj_n  = rng.choice(subj, n, replace=False)
        else:
            subj_n = subj
        print('Plotting data for {} individuals...'.format(len(subj_n)))

        # Sort the individuals in desired order
        if subj_order is not None:
            mask   = np.isin(subj_order, subj_n)
            subj_n = subj_order[mask]

        # Vertical line for diagnosis date
        ax[j].vlines(x=0, ymin=0, ymax=n_vline+vdelta, linestyles='dashed', color='gray', linewidth=0.75, alpha=0.5, zorder=1)

        # Vertical lines marking 6 months
        time_all = df_val.loc[np.isin(df_val.subject, subj_n),timecol]
        if vlines:
            rep = np.floor(np.max(np.abs(time_all))).astype(int)
            x = np.arange(-rep, rep+1)*(365/2)
            x = x[x!=0]
            ax[j].vlines(x=x, ymin=0, ymax=n_vline+vdelta, linestyles='dashed', color='gray', linewidth=0.75, alpha=0.25, zorder=1)

        # Loop over subjects
        for i, s in enumerate(subj_n):

            # Subject's data 
            df_subj = df_val.loc[df_val.subject == s]

            # Get smallest and largest event times
            time   = df_subj[timecol]
            tmax   = time.max()
            tmin   = time.min()
            if tmin > 0:
                tmin = 0
            if tmax < 0:
                tmax = 0

            # Draw horizontal line for patient
            ax[j].hlines(y=i+1, xmin=tmin, xmax=tmax, color='gray', alpha=0.5, linewidth=1, zorder=2)
            
            # Draw treatment period, if requested
            if period:
                pmax = df_subj[periodcol1].iloc[0] + perioddelta
                pmin = df_subj[periodcol0].iloc[0] - perioddelta
                ax[j].hlines(y=i+1, xmin=pmin, xmax=pmax, color='blue', alpha=0.2, linewidth=18, zorder=2)

            # Get event types and plot events in each
            event_types = df_subj['event'].unique()
            for event in event_types:
                time   = df_subj.loc[df_subj.event == event, timecol]
                height = np.ones(len(time))*(i+1)
                color, marker = colors[event][0], colors[event][1]
                ax[j].scatter(time, height, color=color, s=70, alpha=0.5, marker=marker,
                           edgecolors='gray', zorder=3, label=event)   

            # Reformat axes
            ax[j].spines['left'].set_color('gray') #('gray')
            ax[j].spines['right'].set_color('none')
            ax[j].spines['bottom'].set_color('gray')
            ax[j].spines['top'].set_color('none')
            ax[j].set(xlabel=xlabel, ylabel='Subject')
            ax[j].set_ylim(0, n_vline+vdelta)

        # Set x-axis limits
        if (xmin is not None) and (xmax is None):
            ax[j].set_xlim(xmin, time_all.max()+100)
        elif (xmin is None) and (xmax is not None):
            ax[j].set_xlim(time_all.min()-100, xmax)
        elif (xmin is not None) and (xmax is not None):
            ax[j].set_xlim(xmin, xmax)
        else:
            ax[j].set_xlim(time_all.min()-100, time_all.max()+100)
            #ax.set_xticks(np.arange(xmin-2, xmax+1+2, 2))
        if hide_x:
            ax[j].set_xticks([0])
        
        # Force y-axis ticks to be integers
        ax[j].yaxis.set_major_locator(MaxNLocator(integer=True))
        if all_ticks:
            ax[j].yaxis.set_ticks(np.arange(0+1, len(subj_n)+1, 1))
        if subj_axis:
            ax[j].yaxis.set_ticks(np.arange(0+1, len(subj_n)+1, 1))
            ax[j].set_yticklabels(subj_n)
        if hide_y:
            ax[j].set_yticks([0])
            
        # Title
        if title_group is not None:
            ax[j].set_title(label=title_group, #fontdict={'fontweight':'bold'}, 
                            loc=title_loc)
        
        h, l = ax[j].get_legend_handles_labels()
        h, l = np.array(h), np.array(l)
        handles = np.append(handles, h)
        labels  = np.append(labels, l)
    #
    if hide_axis is not None:
        for i in hide_axis:
            ax[i].set_visible(False)

    # Legend
    #handles, labels = ax[j].get_legend_handles_labels()
    #handles, labels = np.array(handles), np.array(labels)
    unique_labels   = event_order[np.isin(event_order, np.unique(labels))]
    idx = [np.where(labels == x)[0][0] for x in unique_labels]
    if n_group > 1:
        if legend_idx is not None:
            for i in legend_idx:
                ax[i].legend(handles[idx], labels[idx], loc='upper right',bbox_to_anchor=(anchor, 1), frameon=False)
        else:
            ax[j].legend(handles[idx], labels[idx], loc='upper right',bbox_to_anchor=(anchor, 1), frameon=False)
    else:
        ax[j].legend(handles[idx], labels[idx], loc='upper right',bbox_to_anchor=(anchor, 1), frameon=False)    
    # Save
    if save:
        if save_pdf:
            plt.savefig(save_path / (plotname+'.pdf'), dpi=dpi, facecolor='white',  bbox_inches='tight', pad_inches=pad_inches)
        else:
            plt.savefig(save_path / (plotname+'.png'), dpi=dpi, facecolor='white',  bbox_inches='tight', pad_inches=pad_inches)
    plt.show()


# ~ Curves over multiple time periods ~
# This could be simplified a lot by re-using a base ROC/PR etc curve
def plot_rocpr_agg(run_path: Path, model_plot: str, periods: list, time_labels: dict, 
                   time_colors: dict, ci: bool = False, figsize: tuple = (18, 6),
                   figsize2: tuple = None, hspace: float = 0.5, vspace: float = 0.33,
                   grid: bool = True, sens_fit10: bool = True, 
                   sens_fit10_alpha: float = 0.75,
                   line_alpha: float = 0.75,
                   models_plot2: list = ['nottingham-lr', 'nottingham-lr-boot', 
                                         'nottingham-cox', 'nottingham-cox-boot', 'fit'],
                   xlow: float = None, suf2: str = "",
                   run_path_all_data: Path = None):
    """Assumes data is in long format, see fitval.metrics.all_metrics"""

    # Output file names
    #region
    suf = '_' + model_plot

    fname = ROCPR_INTERP_PLOT
    if ci:
        fname += '_ci-95'
    else:
        fname += '_ci-none'
    if xlow is not None:
        fname += '_xlim-' + str(xlow)
    fname += suf + '.png'

    fname2 = ROCPR_INTERP_PLOT
    if ci:
        fname2 += '_ci-95'
    else:
        fname2 += '_ci-none'
    if xlow is not None:
        fname2 += '_xlim-' + str(xlow)
    fname2 += suf2 + '_panes.png'

    fname_svg = fname[:-3] + 'svg'
    fname2_svg = fname2[:-3] + 'svg'
    #endregion


    # .... Prepare data ....
    #region

    # Get sensitivity of FIT >= 10
    s = pd.read_csv(run_path / SENS_FIT_CI)
    s = s.loc[(s.model == 'fit') & (s.thr_fit == 10) & (s.metric_name == 'sens')]
    sens_fit = s.set_index('period').metric_value * 100
    sens_fit = sens_fit.rename('sens_fit')
    if run_path_all_data is not None:
        s2 = pd.read_csv(run_path_all_data / SENS_FIT_CI)
        s2 = s2.loc[(s.model == 'fit') & (s2.thr_fit == 10) & (s2.metric_name == 'sens')]
        sens_fit['all'] = s2.metric_value.item() * 100

    # Prepare interpolated ROC and PR data
    roc = pd.read_csv(run_path / ROC_CI)
    if run_path_all_data is not None:
        roc2 = pd.read_csv(run_path_all_data / ROC_CI)
        roc2['period'] = 'all'
        roc = pd.concat(objs=[roc, roc2], axis=0)

    pr = pd.read_csv(run_path / PR_CI)
    if run_path_all_data is not None:
        pr2 = pd.read_csv(run_path_all_data / PR_CI)
        pr2['period'] = 'all'
        pr = pd.concat(objs=[pr, pr2], axis=0)

    roc = roc.loc[roc.metric_name == 'tpr'].rename(columns={'metric_value': 'tpr'})

    sens_max = pr.loc[pr.metric_name=='max_recall'].rename(columns={'metric_value': 'max_recall'})
    sens_max = sens_max.loc[sens_max.model_name=='fit', ['period', 'max_recall']].drop_duplicates()

    pr = pr.loc[pr.metric_name == 'precision'].rename(columns={'metric_value': 'precision'})

    # Prepare interpolated PR gain data
    pr_gain = pd.read_csv(run_path / PR_GAIN_CI)
    if run_path_all_data is not None:
        pr_gain2 = pd.read_csv(run_path_all_data / PR_GAIN_CI)
        pr_gain2['period'] = 'all'
        pr_gain = pd.concat(objs=[pr_gain, pr_gain2], axis=0)

    name = 'proportion_reduction_tests'
    test_red = pr_gain.loc[pr_gain.metric_name == name].rename(columns={'metric_value': name})

    name = 'precision_gain'
    pr_gain = pr_gain.loc[pr_gain.metric_name == name].rename(columns={'metric_value': name})

    # Retain sensitivities below sens_max
    #test_red = test_red.loc[test_red.recall <= sens_max]
    #pr_gain = pr_gain.loc[pr_gain.recall <= sens_max]

    test_red = test_red.merge(sens_max, how='left')
    pr_gain = pr_gain.merge(sens_max, how='left')

    test_red = test_red.loc[test_red.recall <= test_red.max_recall]
    pr_gain = pr_gain.loc[pr_gain.recall <= pr_gain.max_recall]

    # Rescale to percentage
    scale = 100
    roc[['tpr', 'fpr', 'ci_low', 'ci_high']] *= scale
    pr[['recall', 'precision', 'ci_low', 'ci_high']] *= scale
    pr_gain[['recall', 'precision_gain', 'ci_low', 'ci_high']] *= scale
    test_red[['recall', 'proportion_reduction_tests', 'ci_low', 'ci_high']] *= scale
    #endregion


    # .... Plot for single model all time periods ....
    #region
    models_show = [model_plot]

    fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

    for m in models_show:
        for period in periods:
            u = roc.loc[(roc.model_name == m) & (roc.period == period)]
            v = pr.loc[(pr.model_name == m) & (pr.period == period)]
            w = pr_gain.loc[(pr_gain.model_name == m) & (pr_gain.period == period)]
            z = test_red.loc[(test_red.model_name == m) & (test_red.period == period)]

            if m == 'fit':
                plot_line = 'dashed'
                plot_label = None
            else:
                plot_line = 'solid'
                plot_label = time_labels[period]

            # ROC curve
            ax[0].plot(u.fpr, u.tpr, label=plot_label, alpha=line_alpha, color=time_colors[period], linestyle=plot_line)
            #ax[0].fill_betweenx(u.tpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
            if ci:
                ax[0].fill_between(u.fpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=time_colors[period], edgecolor=None)
            if sens_fit10:
                ax[0].axhline(y=sens_fit[period], color=time_colors[period], linestyle='dotted', alpha=sens_fit10_alpha)

            # PR curve
            ax[1].plot(v.recall, v.precision, label=plot_label, alpha=line_alpha, color=time_colors[period], linestyle=plot_line)
            if ci:
                ax[1].fill_between(v.recall, v.ci_low, v.ci_high, alpha=0.2, facecolor=time_colors[period], edgecolor=None)
            if sens_fit10:
                ax[1].axvline(x=sens_fit[period], color=time_colors[period], linestyle='dotted', alpha=sens_fit10_alpha)

            # Gain in PR curve
            if m != 'fit':
                ax[2].plot(z.recall, z.proportion_reduction_tests, label=plot_label, alpha=line_alpha, 
                        color=time_colors[period], linestyle=plot_line)
                if ci:
                    ax[2].fill_between(z.recall, z.ci_low, z.ci_high, alpha=0.2, facecolor=time_colors[period], edgecolor=None)
                if sens_fit10:
                    ax[2].axvline(x=sens_fit[period], color=time_colors[period], linestyle='dotted', alpha=sens_fit10_alpha)

    # Adjust
    ax[0].set(xlabel='False positive rate', ylabel='Sensitivity (% cancers detected)', title='ROC curve')
    #ax[0].legend(frameon=False, loc='best')
    if xlow is None:
        ax[0].set_xticks(np.arange(0, 1.1, 0.1) * scale)
        ax[0].set_yticks(np.arange(0, 1.1, 0.1) * scale)
    else:
        ax[0].set_xticks(np.arange(0, 1.1, 0.1) * scale)
        ax[0].set_yticks(np.arange(xlow, 1.0, 0.05) * scale)  
        ax[0].set(ylim=(xlow * scale, 101))     

    ax[1].set(xlabel='Sensitivity (% cancers detected)', ylabel='Positive predicive value',
             title='Precision-recall curve', ylim=(0, None))
    #ax[1].legend(frameon=False, loc='best')
    if xlow is None:
        ax[1].set_xticks(np.arange(0, 1.1, 0.1) * scale)
        ax[1].set_yticks(np.arange(0, 1.1, 0.1) * scale)
    else:
        ax[1].set_xticks(np.arange(xlow, 1.0, 0.05) * scale)
        ax[1].set_yticks(np.arange(0, 0.55, 0.05) * scale) 
        ax[1].set(xlim=(xlow * scale, None), ylim=(0, 40))           

    ax[2].hlines(y=0, xmin=0, xmax=1*scale, color='red', linestyles='dashed', alpha=0.5, label=model_labels['fit'])
    ax[2].set(xlabel='Sensitivity (% cancers detected)', ylabel='Percent reduction in number of tests (lower is better)',
                title='Test reduction curve')
    ## From Gemini
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[2].get_legend_handles_labels()
    combined_handles = []
    combined_labels = []
    for handle, label in zip(handles1 + handles2, labels1 + labels2):
        if label not in combined_labels or label == "FIT test":
            combined_handles.append(handle)
            combined_labels.append(label)
    ax[2].legend(combined_handles, combined_labels, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    #ax[2].legend(frameon=False, loc='best', bbox_to_anchor=(1.02, 1))
    if xlow is None:
        ax[2].set_xticks(np.arange(0, 1.1, 0.1) * scale)
    else:
        ax[2].set_xticks(np.arange(xlow, 1.0, 0.05) * scale)
        ax[2].set_yticks(np.arange(-0.4, 0.5, 0.1) * scale)    
        ax[2].set(xlim = (xlow * scale, 101), ylim = (-40, 40))   
  
    if grid:
        for z in [0, 1, 2]:
            ax[z].grid(which='major', zorder=1, alpha=0.5)
            #ax[z].minorticks_on()
            #ax[z].grid(which='minor')

    plt.subplots_adjust(hspace=hspace)
    if run_path is not None:
        plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.savefig(run_path / fname_svg, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()
    #endregion


    # .... Plot for all models ....
    #region
    if figsize2 is None:
        figsize2 = (12, len(periods) * 3)
    #fig2, ax_all = plt.subplots(len(periods), 3, figsize=figsize2, constrained_layout=False)
    ## Based on user tdy at https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
    fig2 = plt.figure(constrained_layout=True, figsize=figsize2)
    subfigs = fig2.subfigures(nrows=len(periods), ncols=1)

    abc = string.ascii_uppercase

    model_show = models_plot2
    for i, period in enumerate(periods):

        subfig = subfigs[i]
        subfig.suptitle(abc[i] + '. ' + time_labels[period], x=0.0, ha='left', fontsize=12) #, fontweight='bold')

        #ax = ax_all[i, :]
        ax = subfig.subplots(nrows=1, ncols=3)

        for m in model_show:
            u = roc.loc[(roc.model_name == m) & (roc.period == period)]
            v = pr.loc[(pr.model_name == m) & (pr.period == period)]
            w = pr_gain.loc[(pr_gain.model_name == m) & (pr_gain.period == period)]
            z = test_red.loc[(test_red.model_name == m) & (test_red.period == period)]

            # ROC curve
            ax[0].plot(u.fpr, u.tpr, label=model_labels[m], alpha=0.75, color=model_colors[m])
            #ax[0].fill_betweenx(u.tpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
            if ci:
                ax[0].fill_between(u.fpr, u.ci_low, u.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
            if sens_fit10:
                ax[0].axhline(y=sens_fit[period], color='gray', linestyle='solid', alpha=0.3)

            # PR curve
            ax[1].plot(v.recall, v.precision, label=model_labels[m], alpha=0.75, color=model_colors[m])
            if ci:
                ax[1].fill_between(v.recall, v.ci_low, v.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
            if sens_fit10:
                ax[1].axvline(x=sens_fit[period], color='gray', linestyle='solid', alpha=0.3)

            # Test reduction curve
            if m != 'fit':
                ax[2].plot(z.recall, z.proportion_reduction_tests, label=model_labels[m], alpha=0.75, color=model_colors[m])
                if ci:
                    ax[2].fill_between(z.recall, z.ci_low, z.ci_high, alpha=0.2, facecolor=model_colors[m], edgecolor=None)
                if sens_fit10:
                    ax[2].axvline(x=sens_fit[period], color='gray', linestyle='solid', alpha=0.3)
        
        # Adjust
        ax[0].set(xlabel='False positive rate', ylabel='Sensitivity\n(% cancers detected)', title='ROC curve')
        #ax[0].legend(frameon=False, loc='best')
        if xlow is None:
            ax[0].set_xticks(np.arange(0, 1.1, 0.1) * scale)
            ax[0].set_yticks(np.arange(0, 1.1, 0.1) * scale)
        else:
            ax[0].set_xticks(np.arange(0, 1.1, 0.1) * scale)
            ax[0].set_yticks(np.arange(xlow, 1.0, 0.05) * scale)  
            ax[0].set(ylim=(xlow * scale, 101))  

        ax[1].set(xlabel='Sensitivity\n(% cancers detected)', ylabel='PPV',
                title='Precision-recall curve',
                ylim=(0, None))
        #ax[1].legend(frameon=False, loc='best')
        if xlow is None:
            ax[1].set_xticks(np.arange(0, 1.1, 0.1) * scale)
            ax[1].set_yticks(np.arange(0, 1.1, 0.1) * scale)
        else:
            ax[1].set_xticks(np.arange(xlow, 1.0, 0.05) * scale)
            ax[1].set_yticks(np.arange(0, 0.5, 0.1) * scale) 
            ax[1].set(xlim=(xlow * scale, None), ylim=(0, 40)) 

        ax[2].hlines(y=0, xmin=0, xmax=1*scale, color='red', linestyles='dashed', alpha=0.5, label=model_labels['fit'])
        ax[2].set(xlabel='Sensitivity\n(% cancers detected)', ylabel='Percent reduction in num tests\n(lower is better)',
                title='Test reduction curve')

        ## From Gemini
        handles1, labels1 = ax[0].get_legend_handles_labels()
        handles2, labels2 = ax[2].get_legend_handles_labels()
        combined_handles = []
        combined_labels = []
        for handle, label in zip(handles1 + handles2, labels1 + labels2):
            if label not in combined_labels or label == "FIT test":
                combined_handles.append(handle)
                combined_labels.append(label)
        ax[2].legend(combined_handles, combined_labels, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
        #ax[2].legend(frameon=False, loc='best', bbox_to_anchor=(1.02, 1))
        if xlow is None:
            ax[2].set_xticks(np.arange(0, 1.1, 0.1) * scale)
        else:
            ax[2].set_xticks(np.arange(xlow, 1.0, 0.05) * scale)
            ax[2].set_yticks(np.arange(-0.4, 0.5, 0.1) * scale)    
            ax[2].set(xlim = (xlow * scale, 101), ylim = (-40, 40)) 

        if grid:
            for z in [0, 1, 2]:
                ax[z].grid(which='major', zorder=1, alpha=0.5)
                #ax[z].minorticks_on()
                #ax[z].grid(which='minor')

    #hspace2 = 0.75
    #vspace2 = 0.5
    #plt.subplots_adjust(hspace=hspace2, wspace=vspace2)
    if run_path is not None:
        plt.savefig(run_path / fname2, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.savefig(run_path / fname2_svg, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()

    #endregion
    
    return fig, fig2


def plot_cal_smooth_agg(run_path, model_plot: str, periods: list, time_labels: dict, 
                        time_colors: dict, figsize: tuple = (12, 6),
                        grid: bool = True, figsize2: tuple = None, ci: bool = False,
                        models_incl: list = ['nottingham-lr', 'nottingham-lr-boot', 
                                             'nottingham-cox', 'nottingham-cox-boot'],
                        suf2: str = "", run_path_all_data: Path = None):
    """Assumes data is in long format, see fitval.metrics.all_metrics"""

    # Output file names
    #region
    if model_plot is not None:
        suf = '_' + model_plot

        fname = PLOT_CAL_SMOOTH
        if ci:
            fname += '_ci-95'
        else:
            fname += '_ci-none'
        fname += suf + '.png'

    fname2 = PLOT_CAL_SMOOTH
    if ci:
        fname2 += '_ci-95'
    else:
        fname2 += '_ci-none'
    fname2 += suf2 + '_panes.png'
    
    fname_svg = fname[:-3] + 'svg'
    fname2_svg = fname2[:-3] + 'svg'

    #endregion


    # .... Prepare data ....
    #region

    df = pd.read_csv(run_path / CAL_SMOOTH_CI)
    if run_path_all_data is not None:
        df2 = pd.read_csv(run_path_all_data / CAL_SMOOTH_CI)
        df2['period'] = 'all'
        df = pd.concat(objs=[df, df2], axis=0)
        
    if 'metric_name' in df.columns:
        name = 'prob_true'
        df = df.loc[df.metric_name == name].rename(columns={'metric_value': name})

    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()

    # Spline curves don't have frac, and lowess curves don't have knots 
    # just in case fill in missing values to identify groups correctly
    df.frac = df.frac.fillna('none')
    #df.n_knots = df.n_knots.fillna('none')
    #df.strategy = df.strategy.fillna('none')
    df.fit_ub = df.fit_ub.fillna('none')
    df.loc[df.ymax > 0.2, 'ymax'] = 1  # For file names only
    df.loc[df.ymax < 0.2, 'ymax'] = 0.2   # For file names only

    groups = df[['frac', 'fit_ub', 'ymax']].drop_duplicates()
    assert groups.shape[0] == 2

    groups['title'] = ["A. Clinically relevant range of predicted risks",
                       "B. Full range of predicted risks"]
    groups['title2'] = ["Relevant range of predicted risk",
                       "Full range of predicted risk"]

    #endregion

    # .... Plot for single model over time periods ....
    #region
    if model_plot is not None:
        fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)
        m = model_plot

        for i, (__, row) in enumerate(groups.iterrows()):
            for period in periods:

                mask = (df.frac == row.frac) & (df.fit_ub == row.fit_ub) & (df.ymax == row.ymax)
                mask = mask & (df.period == period) & (df.model_name == m)
                dfsub = df.loc[mask]
                title = row.title

                ax = axs[i]

                # Calibration curve with confidence intervals
                ax.plot(dfsub.prob_pred, dfsub.prob_true, color=time_colors[period], alpha=0.75, 
                        linestyle='solid', label=time_labels[period], zorder=2)
                if ci:
                    ax.fill_between(dfsub.prob_pred, dfsub.ci_low, dfsub.ci_high, alpha=0.25, 
                                    facecolor=time_colors[period], edgecolor=None)

                # Ideal calibration line
                eps = 0.
                ax.plot([0-eps, 1+eps], [0-eps, 1+eps], color='gray', alpha=1, linewidth=1, linestyle='dashed', zorder=1)

                # Adjustments
                ymax = row.ymax
                step = ymax / 10
                xticks = np.arange(0, ymax + step, step)
                if ymax in [0.2, 1]:
                    xlabels = (xticks * 100).astype(int)
                else:
                    xlabels = np.round(xticks * 100, 1)
                
                eps = step / 2
                xlim = (0 - eps, ymax + eps)
                if ymax <= 0.5:
                    yticks = np.arange(0, 0.5 + 0.05, 0.05)
                    ylim=(0 - 0.025, 0.5 + 0.025)
                else:
                    yticks = np.arange(0, 1.1, 0.1)
                    ylim=(0 - eps, 1 + eps)
                ylabels = (yticks * 100).astype(int)

                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                ax.set(xlabel='Predicted probability (%)', ylabel='Observed probability (%)', title=title,
                    xlim=xlim, ylim=ylim)
                ax.legend(frameon=False)

                if grid:
                    ax.grid(which='major', zorder=1, alpha=0.5)
                    #ax.minorticks_on()
                    #ax.grid(which='minor')

        plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
        plt.savefig(run_path / fname_svg, dpi=300, facecolor='white',  bbox_inches='tight')
    #endregion

    # .... Plot multiple models, time periods in rows ....
    if figsize2 is None:
        figsize2 = (10, len(periods) * 3.3)

    fig2 = plt.figure(constrained_layout=True, figsize=figsize2)
    subfigs = fig2.subfigures(nrows=len(periods), ncols=1)

    abc = string.ascii_uppercase
    model_show = models_incl

    for i, period in enumerate(periods):

        subfig = subfigs[i]
        subfig.suptitle(abc[i] + '. ' + time_labels[period], x=0.0, ha='left', fontsize=12) #, fontweight='bold')

        #ax = ax_all[i, :]
        axs = subfig.subplots(nrows=1, ncols=2)
        for i, (__, row) in enumerate(groups.iterrows()):
            ax = axs[i]

            for m in model_show:

                mask = (df.frac == row.frac) & (df.fit_ub == row.fit_ub) & (df.ymax == row.ymax)
                mask = mask & (df.period == period) & (df.model_name == m)
                dfsub = df.loc[mask]
                title = row.title2

                ax = axs[i]

                # Calibration curve with confidence intervals
                ax.plot(dfsub.prob_pred, dfsub.prob_true, color=model_colors[m], alpha=0.75, 
                        linestyle='solid', label=model_labels[m], zorder=2)
                if ci:
                    ax.fill_between(dfsub.prob_pred, dfsub.ci_low, dfsub.ci_high, alpha=0.25, 
                                    facecolor=model_colors[m], edgecolor=None)

                # Ideal calibration line
                eps = 0.
                ax.plot([0-eps, 1+eps], [0-eps, 1+eps], color='gray', alpha=1, linewidth=1, linestyle='dashed', zorder=1)

                # Adjustments
                ymax = row.ymax
                step = ymax / 10
                xticks = np.arange(0, ymax + step, step)
                if ymax in [0.2, 1]:
                    xlabels = (xticks * 100).astype(int)
                else:
                    xlabels = np.round(xticks * 100, 1)
                
                eps = step / 2
                xlim = (0 - eps, ymax + eps)
                if ymax <= 0.5:
                    yticks = np.arange(0, 0.5 + 0.05, 0.05)
                    ylim=(0 - 0.025, 0.5 + 0.025)
                else:
                    yticks = np.arange(0, 1.1, 0.1)
                    ylim=(0 - eps, 1 + eps)
                ylabels = (yticks * 100).astype(int)

                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                ax.set(xlabel='Predicted probability (%)', ylabel='Observed probability (%)', title=title,
                    xlim=xlim, ylim=ylim)
                ax.legend(frameon=False)

            if grid:
                ax.grid(which='major', zorder=1, alpha=0.5)
                #ax.minorticks_on()
                #ax.grid(which='minor')
    plt.savefig(run_path / fname2, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.savefig(run_path / fname2_svg, dpi=300, facecolor='white',  bbox_inches='tight')


def plot_dca_agg(run_path: Path, periods: list, time_labels: dict,
                 model_labels: dict = model_labels, model_colors: dict = model_colors, 
                 figsize: tuple = None, ni: bool = False,
                 models_incl: list = None, ci: bool = False, suf: str = '',
                 run_path_all_data: Path = None):
    print('\nPlotting decision analysis curves...')

    # Output file name
    fname = PLOT_DCA
    if ni:
        fname += '_avoided'
    else:
        fname += '_benefit'

    if ci:
        fname += '_ci-95'
    else:
        fname += '_ci-none'

    fname += suf
    fname += '.png'
    fname_svg = fname[:-3] + 'svg'

    # Data, y-axis value column, y-axis label
    df = pd.read_csv(run_path / DC_CI)
    if run_path_all_data is not None:
        df2 = pd.read_csv(run_path_all_data / DC_CI)
        df2['period'] = 'all'
        df = pd.concat(objs=[df, df2], axis=0)
    if ni:
        col = 'net_intervention_avoided'
        lab = 'Net intervention avoided'
    else:
        col = 'net_benefit'
        lab = 'Net benefit'
    
    df = df.loc[df.metric_name == col].rename(columns={'metric_value': col})
    df = df.sort_values(by=['model_name', 'threshold', col], ascending=[True, True, False])

    # Exclude any models?
    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()

    # Plot
    if figsize is None:
        figsize = (10, len(periods) * 3.3)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(nrows=len(periods), ncols=1)

    abc = string.ascii_uppercase

    for i, period in enumerate(periods):

        subfig = subfigs[i]
        subfig.suptitle(abc[i] + '. ' + time_labels[period], x=0.0, ha='left', fontsize=12) #, fontweight='bold')

        cut = [5, 20, 50]
        axs = subfig.subplots(nrows=1, ncols=len(cut))

        for j, c in enumerate(cut):
            ax = axs[j]
            dfsub = df.loc[(df.threshold <= c / 100) & (df.period == period)]
            title = 'Risk ≤ ' + str(c) + ' %'
            _plot_dca(dfsub, col, lab, models, ax, ci, model_colors, model_labels, title)

        if ni:
            for i in [0, 1, 2]:
                axs[i].legend(frameon=False, loc='lower right')
        else:
            for i in [1, 2]:
                axs[i].legend(frameon=False, loc='upper right')
        axs[0].legend(frameon=False, loc='lower left')


    plt.savefig(run_path / fname, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.savefig(run_path / fname_svg, dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()

