"""Visualise relationships between each variable and linear predictor learned by the models"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from fitval.models import get_model, model_colors, model_labels


variable_xlabels = {"fit_val": "FIT test (μg Hb / g)",
                    "age_at_fit": "Age at FIT (years)",
                    "blood_HGB": "Haemoglobin (g/L)",
                    "blood_PLT": "Platelets (10*9/L)",
                    "blood_MCV": "Mean cell volume (fL)",
                    "blood_CFER": "Serum ferritin (μg/L)",
                    "blood_TRSAT": "Transferrin saturation (%)",
                    "blood_FEN": "Iron (μmol/L)",
                    "ind_gender_M": "Gender (0: F, 1: M)",
                    "irondef": "Iron deficiency (ferritin < 30 μg/L)",
                    "anaemia": "Anaemia (Hb < 110 or 130 g/L)"}


variable_names = {"fit_val": "FIT test",
                  "age_at_fit": "Age at FIT",
                  "blood_HGB": "Haemoglobin",
                  "blood_PLT": "Platelets",
                  "blood_MCV": "Mean cell volume",
                  "blood_CFER": "Serum ferritin",
                  "blood_TRSAT": "Transferrin saturation",
                  "blood_FEN": "Iron",
                  "ind_gender_M": "Gender"}

                  

def _event_rate_per_bin(y: np.ndarray, x: np.ndarray, bins: np.ndarray = None, n_bins: int = 10, strategy: str = 'quantile',
                        bin_cutoff: int = 10):
    """Divide x into bins, return average of x and proportion of events (y==1) in each bin
    Code taken from calibration_curve function in sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
    """

    # Bins
    if bins is None:
        if strategy == "quantile":
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(x, quantiles * 100)
        elif strategy == "uniform":
            bins = np.linspace(x.min(), x.max(), n_bins + 1)
    binids = np.searchsorted(bins[1:-1], x)

    # bin_true: number of observations with y_true = 1 in each bin
    # bin_sums: sum of variable values in each bin
    # bin_total: total number of observations in each bin
    bin_true = np.bincount(binids, weights=y, minlength=len(bins))
    bin_sums = np.bincount(binids, weights=x, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total >= bin_cutoff
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    bin_mean = bin_sums[nonzero] / bin_total[nonzero]
    bin_total = bin_total[nonzero]

    return bin_mean, prob_true, bin_total


def plot_nottingham_small_fit():
    """Illustrate the effects of small FIT values on Nottingham linear predictor"""

    def _linear_predictor_fit(f):
        return -2.19346 * (f/100)**(-1/2) - 0.31620 * (f/100)**(-1/2) * np.log(f/100)

    f = np.arange(0.1, 20, 0.1)
    y = _linear_predictor_fit(f)
    plt.plot(f, y)
    plt.xlabel('FIT value')
    plt.ylabel('Contribution of FIT term')
    plt.show()


def plot_ox(y: np.ndarray, x: pd.DataFrame, ax):

    columns = x.columns.tolist()
    for i, col in enumerate(columns):

        var = x[col].to_numpy()  # Values of variable
        idx = np.where(~np.isnan(var))  # Index of nonmissing values

        # If variable is binary: barplot; else: bins and lowess
        if np.isin(var, [0, 1]).all():
            bin_mean = np.array([0, 1])
            bin_prob = np.array([y[var==0].mean(), y[var==1].mean()])
            ax[i].bar(bin_mean, bin_prob, color='red', alpha=0.75, label=None)
            ax[i].set_xticks([0, 1])

        else:
            # Curves based on binning and LOWESS
            bin_mean, bin_prob, bin_total = _event_rate_per_bin(y[idx], var[idx], n_bins=10, strategy='uniform')
            print('column: {}, bin_totals: {}'.format(col, bin_total))
            lowess = sm.nonparametric.lowess
            z = lowess(y[idx], var[idx], frac=2/3, it=0)
            z[:, 1] = np.clip(z[:, 1], 0, 1)

            ax[i].plot(bin_mean, bin_prob, color='red', alpha=0.75, linestyle='solid', label="Oxford-bins")
            ax[i].scatter(bin_mean, bin_prob, color='red', alpha=0.75, label=None, s=5)
            ax[i].plot(z[:, 0], z[:, 1], color='black', alpha=0.75, linestyle='dashed', label="Oxford-lowess")
        ax[i].set(title=None, ylabel="Probability of CRC", xlabel=variable_xlabels[col])


def plot_model_v2(x: pd.DataFrame, fx: pd.DataFrame, model_name: str, ax, colmap):
    for i, col in colmap.items():
        if col in x.columns:
            var = x[col].to_numpy()  # Values of variable
            pred = fx[col].to_numpy()  # Contribution to linear predictor
            idx = np.where(~np.isnan(var))  # Index of nonmissing values

            # Remove missing values, sort by variable value
            u, v = var[idx], pred[idx]  
            idx_sort = np.argsort(u)
            u, v = u[idx_sort], v[idx_sort]

            # Plot

            ax[i].plot(u, v, color=model_colors[model_name], alpha=0.75, linestyle='solid', label=model_labels[model_name])
            ax[i].set(title=None, ylabel="Add to linear predictor", xlabel=variable_xlabels[col])
            ax[i].legend(frameon=False)


def plot_relations_nottingham_v2(run_path, data_path, lod=True):

    # Read
    cols = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M', 'crc']
    df = pd.read_csv(data_path / 'data_matrix.csv', usecols=cols)

    x = df.drop(labels='crc', axis=1)
    y = df.crc.to_numpy().reshape(-1)

    # Sort columns in same order as in fx returned by model fun 
    x = x[['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M']]

    # Get predictions from model, and model dataframe
    models = ['nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot', 
              'nottingham-fit', 'nottingham-fit-age', 'nottingham-fit-age-sex']
    colmap = {i:col for i, col in enumerate(x.columns)}
    ncol = len(cols) - 1
    nrow = 2 #len(models) + 1
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 5.5, 6), constrained_layout=False)

    plot_ox(y, x, ax[0, :])

    for i, m in enumerate(models):
        model = get_model(m, lod=lod)
        xmod, fx = model(x, return_fx=True, bias_to_fx=False)

        col_reorder = x.columns[x.columns.isin(xmod.columns)]
        xmod = xmod[col_reorder]  # Reorder
        fx = fx[col_reorder]  # Reorder
        plot_model_v2(xmod, fx, m, ax[1,:], colmap)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    fig.savefig(run_path / 'plot_models_nottingham.png', dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()


def plot_nottingham_fit(out_path, plot_type='prob'):
    root = Path("C:\\Users\\1cgQ\\Desktop\\fitval\\") 
    nott_path = root / 'data' / 'nottingham_quantiles.csv'
    ox_path = root / 'results' / 'colofit' / 'data' / 'data_matrix.csv'
    #out_path = root / 'results' / 'fitdist' 

    f = pd.read_csv(nott_path)
    d = pd.read_csv(ox_path, usecols=['fit_val', 'crc'])
    fit_max_ox = d.fit_val.max()

    fit = f.fit_val
    fit = np.linspace(fit.min(), fit.max(), 10000)

    # Visualise contrib of FIT to linear predictor?
    if plot_type == 'lin':
        lin = -2.19346 * (fit / 100)**(-1/2) - 0.31620 * (fit / 100)**(-1/2) * np.log(fit / 100)
        ylab = 'Add to linear predictor'
    
    elif plot_type == 'prob':

        # Visualise predicted probability for median predictor values?
        age = 66
        mcv = 92
        plat = 268
        male = 1

        bias = 0.1216817
        a = 1.96315 * (age / 100)**3 - 15.09326 * (age / 100)**3 * np.log(age / 100)
        f = -2.19346 * (fit / 100)**(-1/2) - 0.31620 * (fit / 100)**(-1/2) * np.log(fit / 100)
        p = 1.07231 * np.log(plat / 100)
        m = -4.73172 * (mcv / 100)
        s = 0.51152 * male
        lin = bias + a + f + p + m + s
        lin = 1 / (1 + np.exp(-lin))
        ylab = 'Predicted probability of CRC'

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(fit, lin, label='Nottingham logistic model')
    ax[0].axvline(x=fit_max_ox, color='red', label='Oxford FIT maximum', alpha=0.75)
    ax[0].set(ylabel=ylab, xlabel='FIT value (µg Hb / g)', title='Full range')
    ax[0].legend(frameon=False)

    ax[1].plot(fit, lin, label='Nottingham logistic model')
    ax[1].axvline(x=fit_max_ox, color='red', label='Oxford FIT maximum', alpha=0.75)
    ax[1].set(ylabel=ylab, xlabel='FIT value (µg Hb / g)', title='Lower range', xlim=[-1, 1001])
    ax[1].legend(frameon=False)
    #tick_values = np.array([0, 10, 100, 400, 1000])
    #ticks = np.log10(tick_values + 1)
    #ax1.set_xticks(ticks)
    #ax1.set_xticklabels(tick_values)
    plt.savefig(out_path / 'plot_fit_model-nottingham-lr.png', dpi=300, facecolor='white',  bbox_inches='tight')
    plt.close()
