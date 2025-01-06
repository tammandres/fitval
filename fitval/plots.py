"""Visualise discrimination and calibration"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from fitval.boot import CAL_SMOOTH_CI, CAL_BIN_NOCI, ROC_NOCI, PR_NOCI, ROC_CI, PR_CI, PR_GAIN_CI, SENS_FIT_CI, DC_CI, SENS_FIT_2_CI


# Output files
PLOT_CAL_BIN = 'plot_calibration_binned'
PLOT_CAL_SMOOTH = 'plot_calibration_smooth'
ROCPR_PLOT = 'plot_rocpr'
ROCPR_INTERP_PLOT = 'plot_rocpr_interp'  # .png and attributes added later depending on plot settings
PLOT_DCA = 'plot_dca'  # .png and attributes added later depending on plot settings


def get_default_colors(models):
    colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    model_colors['fit'] = 'red'
    model_colors['fit-spline'] = 'red'
    return model_colors


# ~ Calibration curves ~
def plot_cal_bin(run_path: Path, model_labels: dict = None,
                 model_colors: dict = None, figsize: tuple = (6, 5), 
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

    # Read data
    df = pd.read_csv(run_path / CAL_BIN_NOCI)
    if 'metric_name' in df.columns:
        name = 'prob_true'
        df = df.loc[df.metric_name == name].rename(columns={'metric_value': name})

    # Model names
    if models_incl is not None:
        models = [mod for mod in models_incl if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()  # Models to be plotted
    
    # Model colors and labels
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}

    # Do not plot bootstrap samples
    if exclude_boot:
        df = df.loc[df['b'] == -1]
    
    # For compatibility with older code
    if 'fit_ub' not in df.columns:
        df['fit_ub'] = 'none'

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


def plot_cal_smooth(run_path: Path, model_labels: dict = None, 
                    model_colors: dict = None, figsize: tuple = (6, 5), grid: bool = False,
                    models_incl: list = None, suf: str = ''):
    """Plot smooth calibration curves"""
    print('\nPlotting smooth calibration curves...')

    df = pd.read_csv(run_path / CAL_SMOOTH_CI)
    if 'metric_name' in df.columns:
        name = 'prob_true'
        df = df.loc[df.metric_name == name].rename(columns={'metric_value': name})
    if 'fit_ub' not in df.columns:
        df['fit_ub'] = 'none'

    if models_incl is not None:
        models = models_incl
        #df = df.loc[df.model_name.isin(models_incl)]
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}

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
def plot_rocpr(run_path: Path, model_labels: dict = None,
               model_colors: dict = None, figsize: tuple = (12, 6), hspace: float = 0.3,
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
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}

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


def plot_rocpr_interp(run_path: Path, model_labels: dict = None, model_colors: dict = None,
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
        s = s.loc[(s.model_name == 'fit') & (s.thr_fit == 10) & (s.metric_name == 'sens_fit')]
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
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}


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
def plot_dca(run_path: Path, model_labels: dict = None, model_colors: dict = None, 
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
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}

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
    all = df.loc[(df.model == 'all')]
    none = df.loc[(df.model == 'none')]
    fit10 = df.loc[(df.model == 'fit10')]
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


def plot_dca_panel(run_path: Path, model_labels: dict = None, model_colors: dict = None, 
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
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}

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


# ~ Reduction in referrals ~
def plot_reduction_tmp(run_path: Path, model_labels: dict = None, model_colors: dict = None, models_incl: list = None):

    # Load data
    df = pd.read_csv(run_path / SENS_FIT_2_CI)

    # x-axis label
    df['xlabel'] = "thr FIT: " + df.thr_fit.astype(str) + "\nthr mod: " + (df.thr_mod * 100).round(4).astype(str)
    df['add_low'] = df.ci_low - df.metric_value
    df['add_high'] = df.ci_high - df.metric_value

    # Quantities to plot, and transform to percentage scale
    red = df.loc[df.metric_name == 'proportion_reduction_tests']
    sens = df.loc[df.metric_name == 'delta_sens']

    red[['metric_value', 'add_low', 'add_high']] *= 100
    sens[['metric_value', 'add_low', 'add_high']] *= 100

    # Model names, colors, labels
    if models_incl is not None:
        models = models_incl
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}
    
    # Plot
    ncol = len(models)
    fig, ax = plt.subplots(2, ncol, figsize=(5 * ncol, 8), constrained_layout=True)
    if ncol == 1:
        ax = [ax]

    for i, model in enumerate(models):
        r = red.loc[red.model_name==model]
        r['xcoord'] = np.arange(r.shape[0])
        ered = r[['add_low', 'add_high']].transpose().abs().to_numpy()
        
        s = sens.loc[sens.model_name==model]
        s['xcoord'] = np.arange(r.shape[0])
        esens = s[['add_low', 'add_high']].transpose().abs().to_numpy()
        
        ax[0, i].errorbar(r.xcoord, r.metric_value, ered, fmt='.', markersize=18, color=model_colors[model])
        ax[0, i].set(title='Reduction in referrals, model: ' + model_labels[model],
                     ylabel='Percent reduction in number of positive tests\nrelative to FIT (negative is better)')
        ax[0, i].hlines(y=0, xmin=0, xmax=s.xcoord.max(), linestyle='dashed', color='red', alpha=0.8)
        ax[0, i].set_xticks(r.xcoord)
        ax[0, i].set_xticklabels(r.xlabel)

        ax[1, i].errorbar(s.xcoord, s.metric_value, esens, fmt='.', markersize=18, color=model_colors[model])
        ax[1, i].set(title='Delta sensitivity, model: ' + model_labels[model],
                     ylabel='Percent cancers missed relative to FIT\n(negative is worse)')
        ax[1, i].hlines(y=0, xmin=0, xmax=s.xcoord.max(), linestyle='dashed', color='red', alpha=0.8)
        ax[1, i].set_xticks(s.xcoord)
        ax[1, i].set_xticklabels(s.xlabel)
    
    plt.savefig(run_path / 'plot_reduction_referrals.png', dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()


def plot_reduction(run_path: Path, model_labels: dict = None, model_colors: dict = None, 
                   models_incl: list = None, thr_fit: float = 10, thr_mod: float = None,
                   figsize: tuple = (12, 6)):

    # Load data
    df = pd.read_csv(run_path / SENS_FIT_2_CI)

    # Filter
    df = df.loc[df.thr_fit == thr_fit]
    assert df.shape[0] > 0

    if thr_mod is not None:
        df = df.loc[df.thr_mod == thr_mod]
        assert df.shape[0] > 0
    else:
        thr_use = df.loc[(df.metric_name=='delta_sens') & (df.metric_value==0), ['model_name', 'thr_mod']]
        df = df.merge(thr_use, how='inner')

    # CI
    df['add_low'] = df.ci_low - df.metric_value
    df['add_high'] = df.ci_high - df.metric_value

    # Quantities to plot, and transform to percentage scale
    red = df.loc[df.metric_name == 'proportion_reduction_tests']
    sens = df.loc[df.metric_name == 'delta_sens']

    red[['metric_value', 'add_low', 'add_high']] *= 100
    sens[['metric_value', 'add_low', 'add_high']] *= 100

    # Model names, colors, labels
    if models_incl is not None:
        models = models_incl
        models = [mod for mod in models if mod in df.model_name.unique()]
    else:
        models = df.model_name.unique()
    
    if model_colors is None:
        model_colors = get_default_colors(models)
    if model_labels is None:
        model_labels = {model:model for model in models}
    
    # Plot
    width = 0.75
    fig, ax = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    
    for i, model in enumerate(models):
        r = red.loc[red.model_name==model]
        err_reduction = r[['add_low', 'add_high']].transpose().abs().to_numpy()
        ax[0].bar(x=[i], height=[r.metric_value.item()], yerr=err_reduction, color=model_colors[model], width=width, label=model_labels[model])
    ax[0].hlines(y=0, xmin=-width/2, xmax=len(models)-1+width/2, linestyle='solid', color='red', alpha=0.8)
    ax[0].set(ylabel='Percent reduction in number of positive tests\nrelative to FIT (negative is better)',
              title='Reduction in referrals')
    ax[0].set_xticks([i for i in range(len(models))])
    ax[0].set_xticklabels(models)
    ax[0].legend(frameon=False, bbox_to_anchor=(1.025, 1), title='Model')

    for i, model in enumerate(models):
        s = sens.loc[sens.model_name==model]
        err_sens = s[['add_low', 'add_high']].transpose().abs().to_numpy()
        ax[1].bar(x=[i], height=[s.metric_value.item()], yerr=err_sens, color=model_colors[model], width=width, label=model_labels[model])
    ax[1].hlines(y=0, xmin=-width/2, xmax=len(models)-1+width/2, linestyle='solid', color='red', alpha=0.8)
    ax[1].set(ylabel='Delta sensitivity (model minus FIT)',
              title='Cancers missed')
    ax[1].set_xticks([i for i in range(len(models))])
    ax[1].set_xticklabels(models)
    # ax[1].legend(frameon=False)

    plt.savefig(run_path / 'plot_reduction_referrals.png', dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()
