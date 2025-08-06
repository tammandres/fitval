"""Compute performance metrics for a prediction model, alone and compared to the FIT test"""
import dcurves as dc
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import statsmodels.api as stat
import warnings
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from statsmodels.stats.proportion import proportion_confint


@dataclass
class PerformanceData:
    """Data class for storing performance metrics"""

    # Global discrimination and calibration metrics
    disc: pd.DataFrame() = pd.DataFrame()  #  Global discrmination metrics
    cal: pd.DataFrame() = pd.DataFrame()  #  Global calibration metrics

    # Basic metrics at various thresholds
    thr_risk: pd.DataFrame() = pd.DataFrame()  # Basic metrics at predefined risk thresholds
    thr_sens: pd.DataFrame() = pd.DataFrame()  # Basic metrics at a small number of predefined sensitivities
    thr_sens_fit: pd.DataFrame() = pd.DataFrame()  # Basic metrics at sensitivities corresponding to FIT thresholds of 2, 10, 100
    thr_sens_fit_gain: pd.DataFrame() = pd.DataFrame()  # Proportion reduction in positive tests compared to FIT
    thr_sens_fit2: pd.DataFrame() = pd.DataFrame()  # Basic metrics at FIT thresholds of 2, 10, 100 and corresponding model_thr estimated in original data
    thr_sens_fit_gain2: pd.DataFrame() = pd.DataFrame()  # Proportion reduction in positive tests compared to FIT, from thr_sens_fit2

    # Receiver-operating characteristic (ROC) curves
    roc: pd.DataFrame() = pd.DataFrame()  # Empirical ROC curve data
    roc_int: pd.DataFrame() = pd.DataFrame()  # Interpolated ROC curve (interpolated sensitivity on fixed grid of fpr)

    # Precision-recall (PR) curves
    pr: pd.DataFrame() = pd.DataFrame()  # Empirical PR curve data
    pr_int: pd.DataFrame() = pd.DataFrame()  # Interpolated PR curve data (PPV, NPV, specificity, TP, FP, TN, FN interpolated at fixed grid of sensitivity)
    pr_gain: pd.DataFrame() = pd.DataFrame()  # Gain in precision compared to FIT, and reduction in number of tests comparaed to FIT, computed from interpolated PR curve data data

    # Calibration curves
    cal_bin: pd.DataFrame() = pd.DataFrame()  # Data for binned calibration curves
    cal_smooth: pd.DataFrame() = pd.DataFrame()  # Data for smooth calibration curves

    # Decision curves
    dc: pd.DataFrame() = pd.DataFrame()  # Data for decision curves, such as net benefit
 

def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, fit: np.ndarray = None, 
                thr_fit: list = [2, 10, 100],
                thr_risk: list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2], 
                sens: list = [0.8, 0.9, 0.95, 0.99],
                interp_step: float = 0.01, ymax_bin: list = [0.2, 1], ymax_lowess: list = [0.2, 1], 
                global_only: bool = False, cal: bool = True, rocpr: bool = True,
                wilson_ci: bool = False, format_long: bool = False,
                raw_rocpr: bool = False, thr_mod: list = None, dca: bool = True):
    """Get performance metrics for a single set of outcome labels and predictions

    Args
        y_true       : true outcome labels (binary), e.g. 0 = no cancer, 1 = cancer
        y_pred       : prediced scores from a model (does not have to be probabilities; must be of same shape as y_true)
        fit          : FIT test values corresponding to y_true (must be of same shape as y_true)
        thr_fit      : FIT threshold to use for evaluation
        thr_risk     : probability thresholds in [0, 1] for computing sensitivity, specificity, PPV, NPV at these thresholds
                       If not specified, a grid of thresholds is created.
        sens         : sensitivity levels in [0, 1] for computing metrics at predefined levels of sensitivity
        interp_step  : interpolation step size for ROC and PR curves
        ymax_bin     : max predicted probability values for generating binned calibration curves
        ymax_lowess  : max predicted probability values for generating LOWESS calibration curves
        global_only  : only compute global metrics (c-statistic, average precision, event rate, mean risk, o/e ratio)
        cal          : if False, no calibration metrics are computed, and no metrics at predefined risk thresholds are computed. 
                       Useful if the model does not return probabilities
        rocpr        : if True, compute ROC and PR curve data
        wilson_ci    : add Wilson CI to some of the computed proportions?
        format_long  : if True, most results are given in format [index col_1, ..., index col_n, metric_name, metric_value]
        raw_rocpr    : if True, empirical ROC and precision recall curves are saved to disk
                       False by default as can take up a lot of disk space
        thr_mod      : model thresholds to use that correspond to FIT thresholds of [2, 10, 100] when computing metrics
                       For example, one could compute thresholds for the model that yield same sensitivities as FIT >= 2, 10 , 100
                       and then include these.
        dca          : Genderate decision curve data?
    
    Returns
        performance metrics in PerformanceData dataclass (see PerformanceData above)
    """

    # Check that inputs are np arrays
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred, np.ndarray):
        raise ValueError("y_pred must be a numpy array")
    if fit is not None:
        if not isinstance(fit, np.ndarray):
            raise ValueError("fit must be a numpy array")
    
    # Ensure all input arrays are 1-dimensional: otherwise can lead to wrong results
    if y_true.ndim > 1 or y_pred.ndim > 1 or fit.ndim > 1:
        raise ValueError("y_true, y_pred and fit must be 1-dimensional numpy arrays. At least one of them is not.")
        
    perf = PerformanceData()

    # ---- Compute global performance metrics ----

    # Global discrimination metrics
    print('... Computing global discrimination metrics')
    perf.disc = global_discrimination_metrics(y_true, y_pred, format_long=format_long)

    # Global calibration metrics
    print('... Computing global calibration metrics')
    if cal:
        perf.cal = global_calibration_metrics(y_true, y_pred, format_long=format_long)
    
    if global_only:
        print('... Only global performance metrics were requested, returning these')
        return perf
    
    # ---- Compute other performance metrics ----

    # Get binary classification curve (needed for multiple computations)?
    #clf_curve = metric_at_observed_thr(y_true, y_pred, add_wilson_ci=False)
    clf_curve = None  # Don't pre-compute atm as have not tested it yet; computed separately
    
    # ROC and PR curve data
    #  Note: when this is bootstrapped, indicates PPV at each level of sensitivity
    #  but the threshold that produces that sensitivity is somewhat different in each sample
    print('... Computing ROC and PR curves')
    if rocpr:
        perf.roc, perf.roc_int, perf.spec_int = \
            roc_data(y_true, y_pred, interp_step, 
                     add_wilson_ci=wilson_ci, sens_add=sens, format_long=format_long,
                     clf_curve=clf_curve)
        perf.pr, perf.pr_int = \
            pr_data(y_true, y_pred, interp_step, 
                    add_wilson_ci=wilson_ci, sens_add=sens, format_long=format_long,
                    clf_curve=clf_curve)
        perf.npv_int = perf.pr_int
    
    if not raw_rocpr: # Remove raw ROC and PR data (can make more elegant later)
        perf.roc = pd.DataFrame()
        perf.pr = pd.DataFrame()

    # Gain in precision and reduction in tests compared to FIT at each level of sensitivity
    # Requires FIT test values to be inputted
    if rocpr and fit is not None:
        print('... Computing PR gain and proportion reduction in tests')
        if format_long:
            pr_int = perf.pr_int
            pr_int = pd.pivot(pr_int, index='recall', columns='metric_name', values='metric_value').reset_index()
        else:
            pr_int = perf.pr_int
        pr_int = pr_int[['recall', 'precision']]
        __, pr_fit = pr_data(y_true, fit, interp_step, add_wilson_ci=wilson_ci, sens_add=sens, format_long=False)
        pr_fit = pr_fit[['recall', 'precision']].rename(columns={'precision': 'precision_fit'})
        pr_gain = pr_int.merge(pr_fit, how='left')
        pr_gain['precision_gain'] = pr_gain.precision - pr_gain.precision_fit
        pr_gain['proportion_reduction_tests'] = pr_gain.precision_fit / pr_gain.precision - 1
        pr_gain['test_ratio_log'] = np.log(pr_gain.proportion_reduction_tests + 1)
        
        if format_long:
            value_cols = ['precision', 'precision_fit', 'precision_gain', 'proportion_reduction_tests',
                          'test_ratio_log']
            pr_gain = pd.melt(pr_gain, id_vars=['recall'], value_vars=value_cols, 
                              var_name='metric_name', value_name='metric_value')
        perf.pr_gain = pr_gain

    # Metrics at sensitivity of FIT >= 10
    #  This computes sens, spec, ppv, npv for FIT >= 10,
    #  and corresponding quantities for each model at the same level of sensitivity as FIT >= 10.
    #  When bootstrapping, the sensitivity will vary over samples (contrary to boostrap CI of PR curve)
    if fit is not None:
        print("Computing metrics at sensitivities of FIT at thresholds", thr_fit)
        perf.thr_sens_fit, perf.thr_sens_fit_gain = \
            metric_at_fit_sens(y_true, y_pred, fit, thr_fit=thr_fit, format_long=format_long)
    
    # Metrics at FIT thresholds thr_fit and corresponding model thresholds thr_mod
    if fit is not None and thr_mod is not None:
        print("Computing metrics at thresholds corresponding to FIT sensitivities at thresholds", thr_fit)
        perf.thr_sens_fit2, perf.thr_sens_fit_gain2 = \
            metric_at_fit_sens(y_true, y_pred, fit, thr_fit=thr_fit, format_long=format_long, thr_mod=thr_mod)

    # Metrics at predefined levels of sensitivity (e.g. 0.8, 0.9, 0.95)
    thr_sens = metric_at_sens(y_true, y_pred, sens, format_long=format_long, clf_curve=clf_curve)
    perf.thr_sens = thr_sens

    # Additional metrics requiring probabilities
    if cal:

        # Sensitivity, specificity, PPV, NPV at risk thresholds
        print('... Computing metrics at thresholds of predicted risk')
        perf.thr_risk = metric_at_risk(y_true, y_pred, thr=thr_risk, format_long=format_long)

        # Decision curve (takes more time to compute than rest of the code)
        if dca:
            print('... Computing decision curves')
            dc = dca_table(y_true, y_pred, format_long=format_long, thr=thr_risk)
            perf.dc = dc

            # Decision curve for FIT >= 10
            if fit is not None:
                print('... Computing decision curves for FIT')
                dc = dca_table(y_true, fit >= 10, model_name='fit10', thr=thr_risk, format_long=format_long)
                perf.dc = pd.concat(objs=[perf.dc, dc], axis=0)

        # Binned calibration curve data
        print('... Computing binned calibration curves with limits', ymax_bin)
        cal_bin = pd.DataFrame()
        for strategy in ['uniform']: #['uniform', 'quantile']:
            for ymax in ymax_bin:
                c = binned_calibration_curve(y_true, y_pred, n_bins=10, ymax=ymax, 
                                             strategy=strategy, format_long=format_long)
                c['fit_ub'] = 'none'
                cal_bin = pd.concat(objs=[cal_bin, c], axis=0)
        
        # Binned calibration curve data for restricted FIT value range
        #if fit is not None:
        #    mask = fit < 200
        #    c = binned_calibration_curve(y_true[mask], y_pred[mask], n_bins=10, ymax=ymax, strategy='uniform')
        #    c['fit_ub'] = '200'
        #    cal_bin = pd.concat(objs=[cal_bin, c], axis=0)
        perf.cal_bin = cal_bin

        # LOWESS calibration curve data
        print('... Computing smooth calibration curves with limits', ymax_lowess)
        cal_smooth = pd.DataFrame()
        for frac in [0.67]: #[0.33, 0.67]:
            for ymax in ymax_lowess:
                c = lowess_calibration_curve(y_true, y_pred, ymax=ymax, frac=frac, format_long=format_long)
                c['fit_ub'] = 'none'
                cal_smooth = pd.concat(objs=[cal_smooth, c], axis=0)

        # LOWESS calibration curve data for restricted FIT value range
        #if fit is not None:
        #    mask = fit < 200
        #    c = lowess_calibration_curve(y_true[mask], y_pred[mask], ymax=1, frac=0.67)
        #    c['fit_ub'] = '200'
        #    cal_smooth = pd.concat(objs=[cal_smooth, c], axis=0)
        perf.cal_smooth = cal_smooth

    return perf


# ~ Global discrimination and calibration metrics ~
#region
def global_discrimination_metrics(y_true: np.ndarray, y_pred: np.ndarray, format_long: bool = True):
    """Compute global discrimination metrics
        auroc: area under ROC curve
        ap: average precision
    """
    # Global discrimination metrics
    auroc = sm.roc_auc_score(y_true, y_pred)
    ap = sm.average_precision_score(y_true, y_pred)
    
    # Gather
    d = {'auroc': auroc, 'ap': ap}
    if format_long:
        r = pd.DataFrame.from_dict(d, orient='index').reset_index()
        r.columns = ['metric_name', 'metric_value']
    else:
        r = pd.DataFrame.from_dict(d, orient='index').transpose()   
    return r


def global_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, format_long: bool = True):
    """Compute global calibration metrics
        event_rate: percent of binary outcomes coded as 1 (e.g. percent of cancers)
        mean_risk: mean predicted risk, percent
        oe_ratio: observed-expected ratio
        log_intercept, log_slope: logistic intercept and slope
    """
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must be in [0, 1]")

    # Global calibration metrics
    event_rate = y_true.mean() * 100
    mean_risk = y_prob.mean() * 100
    oe_ratio = event_rate / mean_risk

    clf = LogisticRegression(penalty=None)
    if (y_prob == 1).any or (y_prob == 0).any():
        warnings.warn('Some predicted probabilities are exactly 0 or 1, replacing with 0 + tol or 1 - tol')
        tol = 1e-6
        y_prob[y_prob == 1] = 1 - tol
        y_prob[y_prob == 0] = 0 + tol

    y_logit = np.log(y_prob / (1 - y_prob))
    clf.fit(y_logit.reshape(-1, 1), y_true)
    log_intercept = clf.intercept_.item()
    log_slope = clf.coef_.item()

    # Gather
    d = {'event_rate': event_rate, 'mean_risk': mean_risk,
         'oe_ratio': oe_ratio, 'log_intercept': log_intercept, 'log_slope': log_slope}
    if format_long:
        r = pd.DataFrame.from_dict(d, orient='index').reset_index()
        r.columns = ['metric_name', 'metric_value']
    else:
        r = pd.DataFrame.from_dict(d, orient='index').transpose()

    return r

#endregion

# ~ Metrics at probability thresholds ~
#region
def dca_table(y_true: np.ndarray, y_prob: np.ndarray, thr: list = None, model_name: str = 'model',
              format_long: bool = False):
    """Table for decision curve analysis"""

    if thr is None:
        #thr = np.linspace(0, 0.2, 41)  # 0.5% increment: 0, 0.005, 0.01, 0.015, ..., 0.195, 0.2
        #thr = np.sort(np.append(thr, [0.006]))  # Add point 0.006 (0.6%)
        #thr = np.unique(np.round(thr, 3))
        thr_low = np.arange(0, 1, 0.05)
        thr_high = np.arange(1, 51, 1)
        thr = np.append(thr_low, thr_high)
        thr = np.sort(np.append(thr, [0.6, 2.5]))  # Add points 0.6% and 2.5%
        thr = thr / 100
        thr = np.unique(np.round(thr, 6))
    else:
        thr = np.array(thr)
        thr = np.unique(np.round(thr, 6))
    
    d = pd.DataFrame(zip(y_true, y_prob), columns=['y_true', model_name])
    r = dc.dca(data=d, outcome='y_true', modelnames=[model_name], thresholds=thr)

    # Add false negative rate
    r['fn_rate'] = r.prevalence - r.tp_rate

    if format_long:
        value_cols = ['n', 'prevalence', 'harm', 'test_pos_rate',
                      'tp_rate', 'fp_rate', 'net_benefit', 'net_intervention_avoided',
                      'fn_rate']
        r = pd.melt(r, id_vars=['model', 'threshold'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    return r


def metric_at_risk(y_true: np.ndarray, y_prob: np.ndarray, thr: list = None, format_long: bool = False,
                   round_digits: int = 6):
    """Compute basic metrics at various risk thresholds.
    Risk means predicted probability according to model.

    Args
        y_true: true outcome labels (binary), e.g. 0 - no cancer, 1 - cancer
        y_prob: predicted probabilities of outcome according to model
        thr: risk thresholds to compute metrics at. If not specified, a grid of thresholds is created
        thr_add: additional risk thresholds to add to the default grid of thresholds
    """

    # If thresholds are not specified, create a fixed grid of risk thresholds
    # The grid covers very low levels of risk from 0 to 1% with 0.05% increments
    # and higher levels of risk from 1% to 100% with 1% increments
    if thr is None:
        #thr = np.array([0, 1, 2, 2.5, 3, 4, 5, 10, 20]) / 100
        thr_low = np.arange(0, 1, 0.05)
        thr_high = np.arange(1, 101, 1)
        thr = np.append(thr_low, thr_high)
        thr = thr / 100
        if round_digits is not None:
            thr = np.round(thr, round_digits)
        thr = np.unique(thr)
    else:
        thr = np.array(thr)
        if round_digits is not None:
            thr = np.round(thr, round_digits)
        thr = np.unique(thr)

    res = pd.DataFrame()
    for t in thr:

        # Apply threshold to predicted probabilities
        test = y_prob >= t
        y_prob_bin = test.astype(int)

        # Get basic metrics at threshold
        m = _basic_metrics(y_true, y_prob_bin, return_counts=False, return_all=True)

        # Get number of positive and negative tests, and positive tests to detect one cancer
        m['pp'] = m.tp + m.fp
        m['pn'] = m.tn + m.fn
        #m['pp_per_cancer'] = np.divide((m.tp + m.fp), m.tp, out=np.full_like(m.tp, np.nan, dtype=np.float64), where=m.tp!=0)
        m['pp_per_cancer'] = np.divide(1, m.ppv, out=np.full_like(m.ppv, np.nan, dtype=np.float64), where=m.ppv!=0)

        # Store
        m = pd.DataFrame([m])
        m = m.astype(float)
        #m = pd.DataFrame(m, columns=['metric_value'])
        #m.index.name = 'metric_name'
        #m = m.reset_index()
        m['thr'] = t
        res = pd.concat(objs=[res, m], axis=0)

    if format_long:
        value_cols = ['sens', 'spec', 'ppv', 'npv', 'tp', 'fp', 'tn', 'fn', 'pp', 'pn', 'pp_per_cancer']
        res = pd.melt(res, id_vars=['thr'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    return res

#endregion

# ~ Calibration curves ~
#region
def binned_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, ymax: int = 1., strategy='uniform',
                             format_long: bool = False):
    """
    Based on calibration_curve from scikit-learn,
    https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98/sklearn/calibration.py#L909
    """
    mask = y_prob > ymax
    if mask.any():
        y_true = y_true[~mask]
        y_prob = y_prob[~mask]

    # Determine bin edges
    if strategy == "quantile": 
        quantiles = np.linspace(0, ymax, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, ymax, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], y_prob)

    # bin_true: number of observations with y_true == 1 in each bin
    # bin_sums: sum of predicted probabilities in each bin
    # bin_total: total number of observations in each bin
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    # Get proportion of events in each nonzero bin (prob_true), 
    # and average predicted prob (prob_pred)
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    # Also get observation and event counts for nonzero bins
    bin_total = bin_total[nonzero]
    bin_true = bin_true[nonzero]
    
    r = pd.DataFrame(zip(prob_pred, prob_true, bin_total, bin_true), 
                     columns=['prob_pred', 'prob_true', 'n_total', 'n_event'])
    r['n_bins'] = n_bins
    r['strategy'] = strategy
    r['ymax'] = ymax

    if format_long:
        value_cols = ['prob_true', 'n_total', 'n_event']
        r = pd.melt(r, id_vars=['prob_pred', 'n_bins', 'strategy', 'ymax'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    return r


def lowess_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, ymax: float = 1., frac: float = 2/3,
                             format_long: bool = False):
    """Data for LOWESS calibration curve"""
    if frac < 0 or frac > 1:
        raise ValueError("frac must be between 0 and 1")

    mask = y_prob > ymax
    if mask.any():
        warnings.warn('ymax is greater than max predicted prob. Removing prob above ymax...')
        y_true = y_true[~mask]
        y_prob = y_prob[~mask]
    
    # Values at which to evaluate
    common_grid = False
    if common_grid:
        x_eval = np.linspace(start=0., stop=1., num=501)  # Create grid with 500 steps
        x_eval = x_eval[x_eval <= ymax]  # Don't evaluate outside ymax
    else:
        x_eval = np.linspace(start=0., stop=ymax, num=100)

    x_eval = np.round(x_eval, 3)
    #x_eval = x_eval[x_eval < y_prob.max()]

    # LOWESS
    lo = stat.nonparametric.lowess
    y_eval = lo(y_true, y_prob, frac=frac, it=0, xvals=x_eval)
    y_eval = np.clip(y_eval, a_min=0, a_max=1)

    r = pd.DataFrame(zip(x_eval, y_eval), columns=['prob_pred', 'prob_true'])
    r['frac'] = frac
    r['ymax'] = ymax

    if format_long:
        value_cols = ['prob_true']
        r = pd.melt(r, id_vars=['prob_pred', 'frac', 'ymax'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    return r

#endregion

# ~ ROC and PR curves ~
#region
def roc_data(y_true: np.ndarray, y_pred: np.ndarray, step: float = 0.025, add_wilson_ci: bool = False,
             sens_add: list = None, format_long: bool = False, interp_spec: bool = False, 
             clf_curve: pd.DataFrame = None):
    """Compute data for receiver-operating characteristic curve"""

    # ROC curve data
    if clf_curve is None:
        fpr, tpr, thr_roc = sm.roc_curve(y_true, y_pred)
        roc = pd.DataFrame(zip(fpr, tpr, thr_roc), columns=['fpr', 'tpr', 'thr'])
        roc['spec'] = 1 - roc.fpr
    else:
        roc = clf_curve
        roc['fpr'] = 1 - roc.spec
        roc = roc.rename(columns={'sens': 'tpr'})

    # Interpolate sensitivity on fixed grid of false positive rate
    roc_int = interpolate_roc(roc, step, target='tpr')

    # Interpolate specificity on fixed grid of sensitivity
    if interp_spec:
        spec_int = interpolate_roc(roc, step, target='fpr', x_add=sens_add)
    else:
        spec_int = pd.DataFrame()

    if format_long:
        value_cols = ['tpr', 'thr']
        roc_int = pd.melt(roc_int, id_vars=['fpr'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

        if interp_spec:
            value_cols = ['fpr', 'spec', 'thr']
            spec_int = pd.melt(spec_int, id_vars=['tpr'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

        value_cols = ['tpr', 'thr']
        roc = pd.melt(roc, id_vars=['fpr', 'spec'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    # Add wilson CI
    if add_wilson_ci and not format_long:
        nobs = len(y_true)
        cols = ['fpr', 'tpr', 'spec']
        for c in cols:
            ci = roc[c].apply(lambda x: proportion_confint(count=nobs*x, nobs=nobs, alpha=0.05, method='wilson'))
            roc[c + '_low'] = ci.apply(lambda x: x[0])
            roc[c + '_high'] = ci.apply(lambda x: x[1])

    return roc, roc_int, spec_int


def pr_data(y_true: np.ndarray, y_pred: np.ndarray, step: float = 0.025, add_wilson_ci: bool = False,
            sens_add: list = None, format_long: bool = False, clf_curve: pd.DataFrame = None):
    """Compute data for precision-recall curve"""

    # Precision-recall curve data
    if clf_curve is None:
        precision, recall, thr_pr = sm.precision_recall_curve(y_true, y_pred)
        thr_pr = np.append(thr_pr, None)
        pr = pd.DataFrame(zip(recall, precision, thr_pr), columns=['recall', 'precision', 'thr'])
    else:
        pr = clf_curve[['sens', 'ppv', 'thr', 'fp', 'tp']]
        pr = pr.rename(columns={'sens': 'recall', 'ppv': 'precision'})

    # Add total number of positives (p), negatives (neg), true positives (tp), false positives (fp)
    pr['p'] = y_true.sum()
    pr['neg'] = (y_true == 0).sum()
    # pr['tp'] = np.round(recall * y_true.sum())
    # pr['fp'] = np.round((1 - precision) * pr['tp'] / precision)
    if clf_curve is None:
        fps, tps, thr = sm._ranking._binary_clf_curve(y_true, y_pred)
        points = pd.DataFrame(zip(fps, tps, thr), columns=['fp', 'tp', 'thr'])
        pr_nrow = pr.shape[0]
        pr = pr.merge(points, how='left', on=['thr'])
        if pr_nrow != pr.shape[0]:
            raise ValueError("Number of rows in PR data has changed after merge, check code for potential rounding errors.")

    # Add false negative and true negative counts
    pr['fn'] = pr.p - pr.tp  
    pr['tn'] = pr.neg - pr.fp  

    # Add NPV and specificity
    pr['npv'] = pr.tn / (pr.fn + pr.tn)
    pr['spec'] = pr.tn / (pr.tn + pr.fp)

    # Interpolate
    pr_int = interpolate_pr(pr, step, sens_add=sens_add)

    if format_long:
        value_cols = ['precision', 'thr', 'fp', 'tp', 'tn', 'fn', 'npv', 'spec', 'max_recall' #,
                      #'p', 'neg', , 'fp_next', 'fp_prev', 'tp_next', 'tp_prev', 'local_skew', 'x', 'bfill',
                     ]
        pr_int = pd.melt(pr_int, id_vars=['recall'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')
        
        value_cols = ['precision', 'thr', 'fp', 'tp', 'tn', 'fn', 'npv', 'spec', 'p', 'neg']
        pr = pd.melt(pr, id_vars=['recall'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    # Add wilson CI
    if add_wilson_ci and not format_long:
        nobs = len(y_true)
        cols = ['recall', 'precision']
        for c in cols:
            ci = pr[c].apply(lambda x: proportion_confint(count=nobs*x, nobs=nobs, alpha=0.05, method='wilson'))
            pr[c + '_low'] = ci.apply(lambda x: x[0])
            pr[c + '_high'] = ci.apply(lambda x: x[1])

    return pr, pr_int


def interpolate_roc(df: pd.DataFrame, step: float = 0.05, target: str = 'tpr', x_add: list = None):
    """Interpolate roc curve data linearly
        df: Pandas DataFrame with columns 'tpr', 'fpr', 'thr'
        step: step size for new tpr (sensitivity) points. If step = 0.05 then new points are [0, 0.05, ..., 0.95, 1]
    
    Threshold is also linearly interpolated, but this for indication only and may not be valid
    """

    x = np.arange(0, 1 + step, step)
    if x_add is not None:
        x = np.append(x, x_add)
    x = np.unique(np.round(x, 6))
    y = [np.nan] * len(x)

    if target == 'fpr':

        # New points
        points = pd.DataFrame(zip(x, y), columns=['tpr', 'fpr'])
        points['interp'] = True

        # Add to dataframe and interpolate
        df = pd.concat(objs=[df, points], axis=0).sort_values(by=['tpr', 'thr'], ascending=[True, False])
        df['fpr'] = df[['fpr']].interpolate('linear')
        df['thr'] = df[['thr']].interpolate('linear')
        df['spec'] = 1 - df.fpr
    
    elif target == 'tpr':

        # New points
        points = pd.DataFrame(zip(x, y), columns=['fpr', 'tpr'])
        points['interp'] = True

        # Add to dataframe and interpolate
        df = pd.concat(objs=[df, points], axis=0).sort_values(by=['fpr', 'thr'], ascending=[True, False])
        df['tpr'] = df[['tpr']].interpolate('linear')
        df['thr'] = df[['thr']].interpolate('linear')
        df['spec'] = 1 - df.fpr

    # Drop old points and return
    return df.loc[df.interp == True].drop(labels=['interp'], axis=1)


def interpolate_pr(df: pd.DataFrame, step: float = 0.05, sens_add: list = None, recall: list = None,
                   interp_thr_nonlin: bool = False):
    """Interpolate precision-recall curve data nonlinearly

        df: Pandas DataFrame with columns
            'recall', 'precision', 'thr', 'tp' (true pos count), 'fp' (false pos count), 
            'p' (total count of positive cases, e.g. num cancers), 
            'neg' (total count of negative cases, e.g. patients without cnacer)
        step: step size for new recall sensitivity) points. If step = 0.05 then new points are [0, 0.05, ..., 0.95, 1]

    Interpolation formula from Davis & Goadrich, 2006, "The relationship between Precision-Recall and ROC curves"

    Key ideas
    * Precision recall curve is based on true positive (TP) and false positive (FP) counts: 
    - precision = TP/(TP + FP)
    - recall = TP/(number of positive cases), where number of positive cases is a constant in the data you are using
    * Let A and B be two points, sorted by increasing recall, and you are interpolating between them
    * At any new point of recall, the number of true positives can simply be calculated as recall * (number of positive cases)
    * Precision depends on how the false positives behave
    - If the number of false positives does not increase between A and B, precision is simply (TPnew) / (TPnew + FP)
    - If the number of false positives increases between A and B, we use the local skew, i.e. the number of false positives per 1 extra true positive

    Edge cases
    * Precisions corresponding to recall = 0 and recall = 1 are not interpolated
    * Precisions that could not be interpolated because they do not lie between two points (e.g. near recall 0), are interpolated using backwards fill
    * If there is no increase in TP between two points, local skew is undefined
    - This should not happen, as PR curve points should always either have increase in FP or increase in TP (though: may need to rigoroushly think through)
      and if they have increase in FP but no increase in TP there is no increase in recall
      and even if the new recall point overlaps with that old recall point, the next point would still have increase in TP
      Local skew is set to 0 just in case it is ever NaN (and for old points, it is)

    Note: threshold is also interpolated, linearly, as an indication of what the threshold may be.
    However, interpolated threshold may not be valid and should be used with caution.
    """

    # Get max sensitivity in data to be interpolated
    # For quality check: it may not be good to interpolate data after that sensitivity
    max_recall = df.loc[df.recall < 1.0].recall.max()

    # New points
    if recall is not None:   ## Only specify a few values of recall
        recall = np.array(recall)
    else:  ## Recall from 0 to 1 at equal steps
        recall = np.arange(0 + step, 1, step)
    
    ## Add specific values of recall to the points to be interpolated IF these not already present in non-interp data
    if sens_add is not None:
        recall = np.append(recall, sens_add)
    recall = np.unique(np.round(recall, 8))
    precision = [np.nan] * len(recall)
    points = pd.DataFrame(zip(recall, precision), columns=['recall', 'precision'])
    points['interp'] = True

    # Add new points to dataframe
    df = pd.concat(objs=[df, points], axis=0).sort_values(by=['recall', 'thr'], ascending=[True, False])

    # p is the total positive count (e.g. number of cancers), neg is the total negative count (e.g. number of non-cancers)
    # they are constants - fill the missing value due to adding new points
    df['p'] = df['p'].fillna(df['p'].iloc[0])
    df['neg'] = df['neg'].fillna(df['neg'].iloc[0])

    # As points are sorted according to increasing recall, get FP and TP counts at the previous and the next position
    # use bfill and ffill to handle multiple consecutive new points
    df['fp_next'] = df.fp.bfill()  #fillna(method='bfill')  # df.fp.shift(-1)
    df['fp_prev'] = df.fp.ffill()  #fillna(method='ffill')  # df.fp.shift(1)
    df['tp_next'] = df.tp.bfill()  #fillna(method='bfill')  # df.tp.shift(-1)
    df['tp_prev'] = df.tp.ffill()  #fillna(method='ffill')  # df.tp.shift(1)
    df['local_skew'] = (df.fp_next - df.fp_prev) / (df.tp_next - df.tp_prev) # Number of false positives per true positive
    df['local_skew'] = df.local_skew.fillna(0)
    df['x'] = df.recall * df.p - df.tp_prev  # Additional true positives gained at the new point of recall

    if interp_thr_nonlin:
        df['thr_next'] = df.thr.bfill()  #fillna(method='bfill')  # df.tp.shift(-1)
        df['thr_prev'] = df.thr.ffill()  #fillna(method='ffill')  # df.tp.shift(1)
        df['local_skew_thr'] = (df.thr_next - df.thr_prev) / (df.tp_next - df.tp_prev) # Number of false positives per true positive
        df['local_skew_thr'] = df.local_skew_thr.fillna(0)

    # Interpolate FP, and get precision
    #  True positive count at the new point is given by tp_prev + x, which is the same as recall * p.
    #  False positive count at the new point is given by num false positives at previous point, plus local_skew * x.
    #  The local_skew indicates how many false positives are incurred per an extra true positive.
    # A note about special cases
    #  If there is no change in True positive count between the previous and next point 
    #    it means that the point to be interpolated already exists in the precision-recall curve. 
    #    In that case, the local_skew will be set to zero, and we use the false positive count 
    #    from the previous point, and the interpolated value will be equal to non-interpolated value.
    #  If there is no change in false positive count between previous and next points,
    #    the local_skew will also be zero and we use the false positive count from the previous point.
    mask = df.interp == True

    tp = df.recall * df.p   # df.tp_prev + df.x  # True positives at the new sensitivity
    fp = df.fp_prev + df.local_skew * df.x  # False positives at the new sensitivity
    precision = tp / (tp + fp)

    df.loc[mask, 'precision'] = precision.loc[mask]
    df.loc[mask, 'tp'] = tp.loc[mask]
    df.loc[mask, 'fp'] = fp.loc[mask]

    if interp_thr_nonlin:
        thr = df.thr_prev + df.local_skew_thr * df.x  # Interpolate threshold
        df.loc[mask, 'thr'] = thr.loc[mask]
        df['thr'] = df.thr.bfill() 

    # Fill poinst near recall = 0 with previous value if they could not be interpolated
    mask = (df.precision.isna()) & (df.interp == True)
    df['bfill'] = 0
    df.loc[mask, 'bfill'] = 1

    df['precision'] = df.precision.bfill()  #fillna(method='bfill')

    df['fp'] = df.fp.bfill()  #fillna(method='bfill')
    df.loc[df.recall == 0, 'fp'] = np.nan

    # Interpolate threshold too -- as an indication where its value may lie
    if not interp_thr_nonlin:
        df['thr'] = df[['thr']].interpolate('linear')

    # Retain only interpolated data
    df = df.loc[df.interp == True]
    df = df.drop(labels=['interp'], axis=1)

    # Add interpolated TN and FN counts, compute npv
    df['fn'] = df.p - df.tp  # False negative count: total positives minus true positives
    df['tn'] = df.neg - df.fp  # True negative count: total negatives minus false positives
    df['npv'] = df.tn / (df.fn + df.tn)

    # Add interpolated specificity too
    df['spec'] = df.tn / (df.tn + df.fp)

    # Add max recall for quality checks
    df['max_recall'] = max_recall

    # For checking the results
    return df

#endregion

# ~ Metrics at selected levels of sensitivity: those corresponding to FIT thresholds, and other ~
#region
def metric_at_fit_sens(y_true: np.ndarray, y_pred: np.ndarray, fit: np.ndarray, 
                       thr_fit: list = [2, 10, 100], model_name: str = None, format_long: bool = False,
                       thr_mod: list = None):
    """Compute common performance metrics for the FIT test at threshold thr_fit
    and for a prediction model at the same level of sensitivity as FIT >= thr_fit

    If thr_mod is specified, the metrics are computed at these thresholds for the model.

    Args
        y_true: true outcome labels (binary), e.g. 0 = no cancer, 1 = cancer
        y_pred: predicted scores according to model (do not have to be probabilities)
        fit: FIT test values (must be same size as y_true and y_pred)
        thr_fit: threshold to apply for the FIT test
        format_long: format results in long format?
        thr_mod: thresholds to apply for the model, corresponding to each threshold in thr_fit
    Output
        out: DataFrame with columns 'model' (model name), 
             'sens' (sensitivity), 'spec' (specificity),
             'ppv' (positive predictive value), 'npv' (negative predictive value),
             'tp' (true positive count), 'fp' (false positive count),
             'tn' (true negative count), 'fn' (false negative count),
             'thr' (threshold), 'interp' (indicates if data for the model was interpolated)
    
    If the performance data of the model does not contain the same level of sensitivity as 
    data of the FIT test (either exactly or up to 4 decmal places), the performance data of the model
    will be interpolated to the same point of sensitivity as the FIT data
    using a method of Davis & Goadrich, 2006, "The relationship between Precision-Recall and ROC curves"
    """
    if thr_mod is not None:
        if not len(thr_fit) == len(thr_mod):
            raise ValueError("If thr_mod is specified, must have as many values as thr_fit")
        thr_mod_use = thr_mod
    else:
        thr_mod_use = [None] * len(thr_fit)

    if model_name is None:
        model_name = 'model'
    
    out = pd.DataFrame()
    out_gain = pd.DataFrame()

    for t, t_mod in zip(thr_fit, thr_mod_use):

        # A. Get performance of FIT at threshold (e.g. FIT >= 10)
        mfit = _basic_metrics(y_true, fit >= t, return_counts=False, return_all=True)
        mfit['thr'] = t
        mfit['interp'] = 0
        mfit['pp'] = mfit.tp + mfit.fp  # Total num of positive tests
        mfit['pn'] = mfit.tn + mfit.fn  # Total num of negative tests
        sens_fit = mfit.sens.item()

        # B. Get performance metrics for the model at same level of sens 
        if thr_mod is None:
            mmod = metric_at_single_sens(y_true, y_pred, sens_fit)
        else:
            mmod = _basic_metrics(y_true, y_pred >= t_mod, return_counts=False, return_all=True)
            mmod['thr'] = t_mod
            mmod['interp'] = 0
            mmod['max_sens'] = np.nan
        mmod['pp'] = mmod.tp + mmod.fp  # Total num of positive tests
        mmod['pn'] = mmod.tn + mmod.fn  # Total num of negative tests

        # Concatenate A and B
        m = pd.DataFrame([mfit, mmod]).reset_index(drop=True)
        m['thr_fit'] = mfit.thr
        m = m.astype(float)
        m['model'] = ['fit', model_name]
        cols_reorder = ['thr_fit', 'model'] + [c for c in m.columns if c not in ['thr_fit', 'model']]
        m = m[cols_reorder]
        out = pd.concat(objs=[out, m], axis=0)

        # Delta A and B
        delta_ppv = mmod.ppv - mfit.ppv
        if thr_mod is None:
            test_red = mfit.ppv / mmod.ppv - 1  # As sensitivity is the same; works better when data was interpolated
        else:
            test_red = (mmod.pp - mfit.pp) / mfit.pp  # As sensitivity is not the same
        test_ratio_log = np.log(test_red + 1)
        delta_sens = mmod.sens - mfit.sens

        m2 = {'thr_fit': mfit.thr, 'model': model_name, 
              'precision_gain': delta_ppv, 'proportion_reduction_tests': test_red, 
              'test_ratio_log': test_ratio_log,
              'delta_sens': delta_sens,
              'ppv_mod': mmod.ppv, 'ppv_fit': mfit.ppv, 
              'pp_mod': mmod.pp, 'pp_fit': mfit.pp,
              'sens_mod': mmod.sens, 'sens_fit': mfit.sens}
        m2 = pd.DataFrame(m2, index=[0])
        out_gain = pd.concat(objs=[out_gain, m2], axis=0)

    # Get number of positive and negative tests, and positive tests to detect one cancer
    out['pp_per_cancer'] = np.divide(1, out.ppv, out=np.full_like(out.ppv, np.nan, dtype=np.float64), where=out.ppv!=0)

    if format_long:
        value_cols = ['sens', 'spec', 'ppv', 'npv', 'tp', 'fp', 'tn', 'fn', 'thr', 'interp', 'max_sens',
                      'pp', 'pn', 'pp_per_cancer'] 
        out = pd.melt(out, id_vars=['thr_fit', 'model'], value_vars=value_cols, 
                     var_name='metric_name', value_name='metric_value')

        value_cols = ['precision_gain', 'proportion_reduction_tests', 'test_ratio_log',
                      'delta_sens',
                      'ppv_mod', 'ppv_fit', 'pp_mod', 'pp_fit', 'sens_mod', 'sens_fit'] 
        out_gain = pd.melt(out_gain, id_vars=['thr_fit', 'model'], value_vars=value_cols, 
                           var_name='metric_name', value_name='metric_value')

    return out, out_gain


def metric_at_sens(y_true: np.ndarray, y_pred: np.ndarray, sens: list = [0.8, 0.9, 0.95, 0.99],
                   format_long: bool = False, clf_curve: pd.DataFrame = None):
    """Specificity, PPV, NPV and other metrics at predefined levels of sensitivity (e.g. 0.8, 0.9, 0.95)
    For a full list of metrics, see metric_at_single_sens
    """
    thr_sens = pd.DataFrame()

    for s in sens:
        print('... Computing metrics at sensitivity', s)
        m = metric_at_single_sens(y_true, y_pred, target_sens=s, clf_curve=clf_curve)
        m = pd.DataFrame([m])
        
        # Get number of positive and negative tests, and positive tests to detect one cancer
        m['pp'] = m.tp + m.fp
        m['pn'] = m.tn + m.fn
        m['pp_per_cancer'] = np.divide(1, m.ppv, out=np.full_like(m.ppv, np.nan, dtype=np.float64), where=m.ppv!=0)

        thr_sens = pd.concat(objs=[m, thr_sens], axis=0)
    thr_sens = thr_sens.reset_index(drop=True)

    if format_long:
        value_cols = ['thr', 'spec', 'ppv', 'npv', 'tp', 'fp', 'tn', 'fn', 'interp', 'max_sens', 'pp', 'pn', 'pp_per_cancer']
        thr_sens = pd.melt(thr_sens, id_vars=['sens'], value_vars=value_cols, var_name='metric_name', value_name='metric_value')

    return thr_sens


def core_metric_at_threshold(y_true, y_pred, fit_val, thr_mod = 0.006, thr_fit = 10, long_format = False):
    """Compute core performance metrics for the model compared to FIT test,
    when model is used at threshold thr_mod and FIT is used at threshold thr_fit 
    """

    pmod = metric_at_risk(y_true, y_pred, thr=[thr_mod], round_digits=None)
    pfit = metric_at_risk(y_true, fit_val, thr=[thr_fit], round_digits=None)
    test_red = ((pmod.pp - pfit.pp) / pfit.pp).item()
    delta_sens = (pmod.sens - pfit.sens).item()

    g = pd.DataFrame({'proportion_reduction_tests': test_red, 
                      'pp_mod_1000': pmod.pp / (pmod.pp + pmod.pn) * 1000,
                      'pp_fit_1000': pfit.pp / (pfit.pp + pfit.pn) * 1000,
                      'delta_sens': delta_sens, 
                      'sens_mod': pmod.sens,
                      'sens_fit': pfit.sens,
                      'thr_mod': thr_mod,
                      'thr_fit': thr_fit,
                      'pp_mod': pmod.pp,
                      'pp_fit': pfit.pp,
                      'n_test': len(y_true),
                      'n_crc': y_true.sum(),
                      'prevalence': y_true.mean()})
    if long_format: 
        g = pd.melt(g, value_name='metric_value', var_name='metric_name')

    return g

#endregion

# ~ Helper functions ~
#region
def _basic_metrics(y_true, y_pred, return_counts=False, return_all=False):
    """Compute basic performance metrics for a single thresholded prediction.

    Args
        y_true: true outcome labels (binary), e.g. 0 = no cancer, 1 = cancer
        y_pred: predicted scores (thresholded, binary)

    Returns
        If return_counts is True and return_all is False, returns TP, FP, TN, FN counts
        If return_counts is False and return_all is False, returns sensitivity, specficity, PPV, NPV
        If return counts is False and return_all is True, returns
        sensitivity, specficity, PPV, NPV, TP, FP, TN, FN 
    """
    if return_counts and return_all:
        raise ValueError("return_counts and return_all must not be True at the same time")
    
    # Get tp, fp, tn, fn counts
    tn, fp, fn, tp = sm.confusion_matrix(y_true, y_pred).ravel()

    # To numpy array and float, allowing quantities computed below to be nan in edge cases
    tp, fp, tn, fn = [np.array(a).astype(float) for a in [tp, fp, tn, fn]]  

    # Compute sensitivity, specificity, PPV, NPV
    sens = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=np.float64), where=(tp + fn) != 0)
    ppv = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan, dtype=np.float64), where=(tp + fp) != 0)
    spec = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan, dtype=np.float64), where=(tn + fp) != 0)
    npv = np.divide(tn, tn + fn, out=np.full_like(tn, np.nan, dtype=np.float64), where=(tn + fn) != 0)

    if return_counts:
        return tp, fp, tn, fn
    elif return_all:
        out = pd.Series([sens, spec, ppv, npv, tp, fp, tn, fn], 
                            index=['sens', 'spec', 'ppv', 'npv', 'tp', 'fp', 'tn', 'fn'])
        return out
    else:
        return sens, spec, ppv, npv


def metric_at_observed_thr(y_true: np.ndarray, y_pred: np.ndarray, add_wilson_ci: bool = False):
    """Compute basic performance metrics at all possible binary classification thresholds.
    ROC and PR curve data as computed by other scikit-learn functions is also based on that.

    Args
        y_true : true outcome labels (binary), e.g. 0 = no cancer, 1 = cancer
        y_pred : predicted scores (do not have to be probabilities)
        add_wilson_ci : whether to add Wilson CI for proportions
    
    Returns
        m : DataFrame with columns 'thr' (threshold used), 'sens' (sensitivity), 
            'spec' (specificity), 'ppv' (positive predictive value), 'npv' (negative predictive value),
            'tp', 'fp', 'tn', 'fn' (true pos, false pos, true neg, false neg counts),
            'interp' (indicator for whether data was interpolated to compute metrics at target sensitivity)

    For more information, see 
    https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/metrics/_ranking.py#L712
    """

    # FP, TP, FN, TN at each threshold
    fp, tp, thr = sm._ranking._binary_clf_curve(y_true, y_pred)
    fn = len(y_true[y_true == 1]) - tp  # False negative count: total positives minus true positives
    tn = len(y_true[y_true == 0]) - fp  # True negative count: total negatives minus false positives

    # Compute sens, ppv, spec, npv; allowing it to be nan in edge cases, e.g. when tp+fp = 0
    sens = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=np.float64), where=(tp + fn) != 0)
    ppv = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan, dtype=np.float64), where=(tp + fp) != 0)
    spec = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan, dtype=np.float64), where=(tn + fp) != 0)
    npv = np.divide(tn, tn + fn, out=np.full_like(tn, np.nan, dtype=np.float64), where=(tn + fn) != 0)

    # Gather
    m = {'thr': thr, 'sens': sens, 'spec': spec, 'ppv': ppv, 'npv': npv, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    m = pd.DataFrame.from_dict(m)

    # Add Wilson CI?
    if add_wilson_ci:
        nobs = len(y_true)
        cols = ['sens', 'spec', 'ppv', 'npv']
        for c in cols:
            ci = m[c].apply(lambda x: proportion_confint(count=nobs*x, nobs=nobs, alpha=0.05, method='wilson'))
            m[c + '_low'] = ci.apply(lambda x: x[0])
            m[c + '_high'] = ci.apply(lambda x: x[1])
    
    return m


def metric_at_single_sens(y_true: np.ndarray, y_pred: np.ndarray, target_sens: float, round_digits: int = 4,
                          clf_curve: pd.DataFrame = None):
    """Compute common performance metrics at a single predefined level of sensitivity.

    Args
        y_true : true outcome labels (binary), e.g. 0 = no cancer, 1 = cancer
        y_pred : predicted scores (do not have to be probabilities)
        target_sens : target sensitivity as a fraction, e.g. 0.90; numpy array 

    Returns
        mmod : pd.Series with values 'thr' (threshold used), 'sens' (sensitivity),
               'spec' (specificity), 'ppv' (positive predictive value), 'npv' (negative predictive value),
               'tp', 'fp', 'tn', 'fn' (true pos, false pos, true neg, false neg counts),
               'interp' (indicator for whether data was interpolated to compute metrics at target sensitivity)
    
    Note
        If the goal is to calculate metrics at multiple target sensitivities, 
        it can be more convenient to use the pr_data function.
    """

    # Compute metrics at same level of sens as thresholded FIT
    if clf_curve is None:
        m = metric_at_observed_thr(y_true, y_pred, add_wilson_ci=False)
    else:
        m = clf_curve
    max_sens = m.loc[m.sens < 1.0].sens.max() # Get largest sensitivity that is not 1 (not good to interpolate sens greater than that)

    # If performance data of model has the same sensitivity, use this
    test = m.sens == target_sens
    if test.any():  
        mmod = m.loc[test].sort_values(by='fp', ascending=True).iloc[0]
        mmod['interp'] = 0
    else:  
        # If performance data of model has the same sensitivity up to 4 decimal points, use this
        test2 = m.sens.round(round_digits) == np.round(target_sens, round_digits)
        if test2.any():  
            mmod = m.loc[test2].sort_values(by='fp', ascending=True).iloc[0]
            mmod['interp'] = 0
        
        # If performance data of model does not have the same sensitivity up to 4 decimal points: interpolate
        else:   
            sens_new = np.round(target_sens, round_digits)
            m_tmp = m.rename(columns={'sens': 'recall', 'ppv': 'precision'})
            m_tmp['p'] = m_tmp.tp + m_tmp.fn
            m_tmp['neg'] = m_tmp.tn + m_tmp.fp
            mmod = interpolate_pr(df=m_tmp, recall=[sens_new])
            mmod = mmod.rename(columns={'recall': 'sens', 'precision': 'ppv'})
            mmod = mmod[['thr', 'sens', 'spec', 'ppv', 'npv', 'tp', 'fp', 'tn', 'fn']]
            mmod['interp'] = 1
            mmod = mmod.squeeze()

    mmod['max_sens'] = max_sens
    mmod.name = 0
    return mmod

#endregion

# Breathing in, breathing out, checking in with the body