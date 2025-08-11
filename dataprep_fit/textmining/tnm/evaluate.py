"""Compare extracted vs ground-truth TNM scores"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.proportion import proportion_confint
from textmining.constants import RESULTS_DIR


TNM_RESULTS = 'performance_tnm.csv'
TNM_RESULTS_DETAILED = 'performance_tnm_detailed.csv'


def evaluate_tnm(truth_path: Path, eval_path: Path):
    print('\nComparing ground-truth TNM values to predicted TNM values...')

    # Columns to be compared
    cols_max = ['T', 'N', 'M', 'V', 'R', 'L', 'Pn', 'SM', 'H', 'G']
    cols_min = [c + '_min' for c in cols_max]
    cols = cols_max + cols_min

    # Load and prepare data for comparison
    df0, df1 = _prepare_data(truth_path, eval_path, cols)

    # Compute performance for each TNM category (e.g. T and N), separately for each report type
    res = pd.DataFrame()
    for rtype in df0.report_type.unique():
        for c in cols:
            if rtype == 'imaging' and c in ['R', 'L', 'Pn']:  # Exclude columns not relevant for imaging
                continue
            else:
                c0 = df0.loc[df0.report_type == rtype, c]
                c1 = df1.loc[df1.report_type == rtype, c]
                r = _accuracy(c0, c1, digits=2, ci=True, ci_method='wilson')
                r['tnm_category'] = c
                r['report_type'] = rtype
                res = pd.concat(objs=[res, r], axis=0)
    res = res[['report_type', 'tnm_category', 'n_total', 'n_notnull', 'n_null', 'accuracy', 'balanced_accuracy',
               'sensitivity_any', 'specificity_any']]

    # Compute performance for each value of each TNM category, separately for each report type
    # For example, T category may take values ('x', '0', '1', '2'),
    # and performance is computed separately for each type of value
    cat = pd.DataFrame()
    for rtype in df0.report_type.unique():
        for c in cols:
            if rtype == 'imaging' and c in ['R', 'L', 'Pn']:  # Exclude columns not relevant for imaging
                continue
            else:
                c0 = df0.loc[df0.report_type == rtype, c]
                c1 = df1.loc[df1.report_type == rtype, c]
                vals = np.unique(c0)
                for v in vals:
                    v0 = c0[c0 == v]
                    v1 = c1[c0 == v]
                    r = _accuracy(v0, v1, digits=2, ci=True)
                    r['tnm_category'] = c
                    r['tnm_subcategory'] = v
                    r['report_type'] = rtype
                    cat = pd.concat(objs=[cat, r], axis=0)
    cat = cat[['report_type', 'tnm_category', 'tnm_subcategory', 'n_total', 'n_notnull', 'n_null', 'accuracy',
               'balanced_accuracy', 'sensitivity_any', 'specificity_any']]

    # Save
    res.to_csv(RESULTS_DIR / TNM_RESULTS, index=False)
    cat.to_csv(RESULTS_DIR / TNM_RESULTS_DETAILED, index=False)

    return res, cat


def _prepare_data(truth_path, eval_path, cols):

    # Read ground truth data (0), and data to be evaluated (1)
    df0 = pd.read_csv(truth_path)
    df1 = pd.read_csv(eval_path)

    # Check that columns to be compared are present
    test0 = np.isin(cols, df0.columns).all()
    test1 = np.isin(cols, df1.columns).all()
    if not test0 and test1:
        raise ValueError('Not all columns to be evaluated are present in datasets')

    # Check that indices match
    test = np.all(df0.index == df1.index)
    if not test:
        raise ValueError("Indices of ground truth data, and data to be evaluated, don't match")

    # Check that 'report_type' is same between data sets, if present
    if 'report_type' in df0.columns:
        test = np.all(df0.report_type == df1.report_type)
        if not test:
            raise ValueError('report_type not same in ground truth data, and data to be evaluated')

    # Replace missing values and empty strings with 'null', and convert all values to lowercase strings
    for df in [df0, df1]:
        for c in cols:
            df[c] = df[c].fillna('null')
            df[c] = df[c].astype(str)
            df[c] = df[c].str.lower()
            df[c] = df[c].str.replace(r'\s{1,}', 'null', regex=True)
        nmis = df[cols].isna().sum().sum()
        if nmis > 0:
            raise ValueError('Not all missing values were replaced')

        # Add report_type column if not present
        if 'report_type' not in df.columns:
            warnings.warn("Adding 'report_type' column")
            df['report_type'] = 'none'

    return df0, df1


def _accuracy(y0: np.array, y1: np.array, digits: int = 2, ci: bool = True, ci_method: str = 'wilson'):
    """Compute accuracy, balanced accuracy, sensitivity, specificity

    Args:
        y0: ground truth data
        y1: predicted data

    Note: balanced accuracy is the average of recall (sensitivity) obtained for each class.
        Suppose one of the columns to be compared is called 'T' (T-score),
        and that it has four unique values in the datat: ['null', '1', '1a', 'x'].
        Then balanced accuracy is the average of sensitivities for 'null', '1', '1a', 'x'.
        It is high only if all of these types of values are detected correctly.
        This helps with class imbalance: the performance estimate won't be driven by the majority classes.

        Note that if 'null' means that no value is present, then sensitivity for 'null' values
        is the same as specificity for detecting that no value was present.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """

    # Compare
    comp = y0 == y1

    # Accuracy and balanced accuracy
    n_total = comp.shape[0]
    acc = comp.sum(axis=0) / n_total * 100
    bacc = balanced_accuracy_score(y0, y1) * 100

    # Sensitivity for detecting any TNM value
    #  For example, if a report contains any T-value (such as '1', '2', 'x', '3', 'is' etc),
    #  what is the probability that it is marked as containing the correct T-value?
    mask = y0 != 'null'
    num = mask.sum(axis=0)
    c = comp.copy()
    c[~mask] = False
    sens = c.sum(axis=0) / num * 100
    n_notnull = num

    # Specificity for detecting a TNM value
    #  For example, if a report does not contain any T-values,
    #  what is the probability that it is marked as not containing a T-value?
    mask = y0 == 'null'
    num = mask.sum(axis=0)
    c = comp.copy()
    c[~mask] = False
    spec = c.sum(axis=0) / num * 100
    n_null = num

    # Combine
    res = pd.DataFrame([[n_total, n_notnull, n_null, acc, bacc, sens, spec]],
                       columns=['n_total', 'n_notnull', 'n_null', 'accuracy', 'balanced_accuracy', 'sensitivity_any',
                                'specificity_any'])

    # CI
    if ci:
        for col, nobs in zip(['accuracy', 'sensitivity_any', 'specificity_any'], [n_total, n_notnull, n_null]):
            count = res[col] / 100 * nobs
            conf = proportion_confint(count=count, nobs=nobs, alpha=0.05, method=ci_method)
            low = (conf[0] * 100).round(digits).astype(str)
            high = (conf[1] * 100).round(digits).astype(str)
            res[col] = res[col].round(digits).astype(str) + ' (' + low + ', ' + high + ')'
            res.loc[conf[0].isna(), col] = np.nan

        for col in ['balanced_accuracy']:
            res[col] = res[col].round(digits)
    else:
        for col in ['accuracy', 'balanced_accuracy', 'sensitivity_any', 'specificity_any']:
            res[col] = res[col].round(digits)

    return res


# ==== More vectorised accuracy function (not used atm) ====
def _accuracy_all(df0, df1, cols, digits=2, ci=True, ci_method='wilson'):
    """Compute accuracy, balanced accuracy, sensitivity, specificity
    Args:
        df0: ground truth data
        df1: data to be evaluated
        cols: columns to be compared

    Note: balanced accuracy is the average of recall (sensitivity) obtained for each class.
        Suppose one of the columns to be compared is called 'T' (T-score),
        and that it has four unique values in the datat: ['null', '1', '1a', 'x'].
        Then balanced accuracy is the average of sensitivities for 'null', '1', '1a', 'x'.
        It is high only if all of these types of values are detected correctly.
        This helps with class imbalance: the performance estimate won't be driven by the majority classes.

        Note that if 'null' means that no value is present, then sensitivity for 'null' values
        is the same as specificity for detecting that no value was present.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """

    # Compare
    y0 = df0[cols]
    y1 = df1[cols]
    comp = y0 == y1

    # Accuracy
    n_total = comp.shape[0]
    acc = comp.sum(axis=0) / n_total * 100
    acc = pd.DataFrame(acc.rename('accuracy'))

    # Balanced accuracy
    bacc = pd.Series([balanced_accuracy_score(y0[c], y1[c]) * 100 for c in cols], index=cols, name='balanced_accuracy')

    # Sensitivity for detecting any TNM value
    #  For example, if a report contains any T-value (such as '1', '2', 'x', '3', 'is' etc),
    #  what is the probability that it is marked as containing the correct T-value?
    mask = y0 != 'null'
    num = mask.sum(axis=0).rename('n_notnull')
    c = comp.copy()
    c[~mask] = False
    sens = c.sum(axis=0) / num * 100
    sens = sens.rename('sensitivity_any')
    n_notnull = num

    # Specificity for detecting a TNM value
    #  For example, if a report does not contain any T-values,
    #  what is the probability that it is marked as not containing a T-value?
    mask = y0 == 'null'
    num = mask.sum(axis=0).rename('n_null')
    c = comp.copy()
    c[~mask] = False
    spec = c.sum(axis=0) / num * 100
    spec = spec.rename('specificity_any')
    n_null = num

    # Combine
    res = pd.concat(objs=[acc, bacc, sens, spec], axis=1)
    res['n_total'] = n_total
    res['n_notnull'] = n_notnull
    res['n_null'] = n_null
    res = res[['n_total', 'n_notnull', 'n_null', 'accuracy', 'balanced_accuracy', 'sensitivity_any', 'specificity_any']]

    # CI
    if ci:
        for col, nobs in zip(['accuracy', 'sensitivity_any', 'specificity_any'], [n_total, n_notnull, n_null]):
            count = res[col] / 100 * nobs
            conf = proportion_confint(count=count, nobs=nobs, alpha=0.05, method=ci_method)
            low = (conf[0] * 100).round(digits).astype(str)
            high = (conf[1] * 100).round(digits).astype(str)
            res[col] = res[col].round(digits).astype(str) + ' (' + low + ', ' + high + ')'
            res.loc[conf[0].isna(), col] = np.nan

        for col in ['balanced_accuracy']:
            res[col] = res[col].round(digits)
    else:
        for col in ['accuracy', 'balanced_accuracy', 'sensitivity_any', 'specificity_any']:
            res[col] = res[col].round(digits)

    # Reset index
    res = res.reset_index().rename(columns={'index': 'tnm_category'})

    return res
