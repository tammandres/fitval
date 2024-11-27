"""Apply prediction models to data and compute metrics with bootstrap confidence intervals,
imputing the data in each bootstrap sample if there are missing values

Process with imputation
1. Take a bootstrap sample of the original data
2. Impute the sample M times
3. Apply prediction model(s) to each of the M imputed datasets
4. Compute performance metrics in each of the M imputed datasets
5. Average performance metrics over M datasets
6. Repeat steps 1-5 B times to obtain a bootstrap distribution for each performance metric
7. Use bootstrap percentile method to get confidence intervals for the performance metric

Process without imputation
1. Take a bootstrap sample of the original data
2. Apply prediction model(s) to the sample (or use existing model predictions)
3. Compute performance metrics
4. Repeat steps 1-5 B times to obtain a bootstrap distribution for each performance metric
5. Use bootstrap percentile method to get confidence intervals for the performance metric

Note:
* In some bootstrap samples, metrics for FIT test can be nan at low sensitivities (e.g. 0.5).
  For example, it may be that at the highest FIT threshold, sensitivity is 53%, 
  so there are no lower sensitivities.
"""
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from joblib import Parallel, delayed
from pathlib import Path
from fitval.metrics import PerformanceData, all_metrics, metric_at_single_sens
from fitval.models import get_model, create_spline_model
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress pandas future warnings (for dcurves package)
#warnings.simplefilter(action='ignore', category=UserWarning)  # Suppress UserWarning


# Output files for metrics (CI: bootstrap CI is computed; NOCI: no bootstrap CI is computed)

## Overall discrimination and calibration metrics
DISC_CI = 'discrimination.csv'
CAL_CI = 'calibration.csv'

## Calibration curves
CAL_SMOOTH_CI = 'calibration_curve_smooth.csv'
CAL_BIN_NOCI = 'calibration_curve_binned.csv'

## Metrics at risk score thresholds, and sensitivities
RISK_CI = 'metrics_at_risk.csv'
SENS_CI = 'metrics_at_sens.csv'

## Metrics at ...
SENS_FIT_CI = 'metrics_at_sens_fit.csv'
SENS_FIT_2_CI = 'metrics_at_fit_and_mod_thresholds.csv'

## ROC curve data
ROC_CI = 'roc_interp.csv'
ROC_NOCI = 'roc.csv'

## PR curve data
PR_CI = 'pr_interp.csv'
PR_GAIN_CI = 'pr_gain.csv'
PR_NOCI = 'pr.csv'

## Decision curve data
DC_CI = 'dc.csv'


def boot_metrics(data_path: Path,
                 save_path: Path = None,
                 data_has_predictions: bool = True,
                 model_names: list = None,
                 recal: bool = True,
                 thr_risk: list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2], 
                 sens: list = [0.5, 0.8, 0.85, 0.9, 0.95, 0.99], 
                 thr_fit: list = [2, 10],
                 prob_min: float = 0.2,
                 B: int = 500, 
                 boot_method: str = 'percentile',
                 M: int = 10,
                 max_iter: int = 10,
                 impute_with_y: bool = False,
                 random_state: int = 42,
                 n_noci: int = 5, 
                 parallel: bool = False,
                 nchunks: int = 15, 
                 raw_rocpr: bool = False, 
                 plot_boot: bool = True,
                 return_boot_samples: bool = False
                 ):
    """Compute performance metrics for predefined models on dataset (x, y), 
    and obtain bootstrap confidence intervals for the metrics

    Args
        data_path: Path to a .csv file that contains the input data
        save_path: Path to a directory where outputs should be saved
        data_has_predictions: 
            If True, the input data must have columns "y_true", "y_pred" and "fit_val",
                where y_true contains 0-1 outcome labels (e.g. 0: no cancer, 1: cancer),
                y_pred contains predicted probabilities of the outcome according to a model, and
                fit_val contains numerical FIT test values.
                If there are multiple models, their predictions must be named as 'y_pred1', 'y_pred2', 'y_pred3' etc.
            If False, the input data must have columns "y_true" and "fit_val" as defined above,
                and additional columns required to apply a prediction model.
        model_names: names of models that were used to compute predictions, for example ['logistic-full', 'logistic-restricted'].
                If data_has_predictions is True, there must be as many names as there are columns that start with name 'y_pred'.
        recal: apply logistic recalibration to all models? In that case, recalibrated models are evaluated alongside the original ones.
        thr_risk: thresholds of predicted risk to compute metrics for
        sens: sensitivity values at which to compute metrics
        prob_min: x-axis limit for predicted probabilities in calibration curves that show predicted risks in lower range
        B: number of bootstrap samples
        boot_method: method of computing bootstrap CIs, either 'percentile' or 'basic'
        M: number of multiple imputation rounds
        max_iter: maximum number of iterations at each of the multiple imputation rounds
        impute_with_y: include outcome variable in imputation models?
        random_state: initial random state
        n_noci: number of bootstrap samples to save for metrics where bootstrap CIs cannot be computed (e.g. raw ROC and PR curves)
                as the raw ROC curve data is large, saving all bootstrap samples is not good
        parallel: process bootstrap samples using parallel cores
        nchunks: number of chunks to divide the B bootstrap samples into for parallel processing
        raw_rocpr: do not save raw ROC and PR curves (can take up disk space)
        plot_boot: plot bootstrap distributions of some metrics?
        return_boot_samples: if True, a dataclass containing bootstrap samples is returned.
    
    Returns
        if return_boot_samples is False:
            ci_data: PerformanceData class (see fitval/metrics.py) that contains metrics along with bootstrap confidence intervals.
                    Fields for which bootstrap CIs cannot be computed are empty and stored in noci_data instead
            noci_data: dataclass that contains metrics for which bootstrap CIs cannot be computed.
                    .roc: ROC curve data along with n_noci bootstrap samples
                    .pr: PR curve data with n_noci bootstrap samples
                    .cal_bin: binned calibration curve data with n_noci bootstrap samples 
        otherwise it returns a PerformanceData class (see fitval/metrics.py) that contains metrics along with bootstrap samples.
    """
    tic = time.time()
    
    # Additional arguments that usually do not need to be changed
    global_only = False  # If True, only global discrimination and calibration metrics are computed
    interp_step = 0.01  # Interpolation step for PR and ROC curves, 0.01 means 1% increment
    stratified_boot = False  # stratified bootstrap (bootstrap samples taken separately for y==0 and y==1 retaining the proportion of y==1)
    repl_ypred_nan = False  # if a model returns a nan for prediction, replace it with zero? False by default, only used for testing
    fit_spline = True  # Should the fitted FIT-only spline model be subsequently evaluated?
    plot_fit_model = True

    # Argument validation
    if not isinstance(data_path, Path):
        raise TypeError(f"Expected Path object for data_path, got {type(data_path)}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")
    
    if save_path is not None:
        if not isinstance(save_path, Path):
            raise TypeError(f"Expected Path object for save_path, got {type(save_path)}")
        if not save_path.exists():
            raise FileNotFoundError(f"save_path does not exist: {save_path}")
    
    if not data_has_predictions and model_names is None:
        raise ValueError("Model names must be specified if data_has_predictions is False")
    
    # Adjust number of chunks (only relevant if parallel = True)
    if B < nchunks:
        print("Number of bootstrap samples B is smaller than number of chunks, setting nchunks=B")
        nchunks = B
    
    # Read data
    df = pd.read_csv(data_path)

    # Validate data columns
    target_cols = np.array(['y_true', 'fit_val'])
    test = [c not in df.columns for c in target_cols]
    if any(test):
        raise ValueError("Column(s) " + str(target_cols[test].tolist()) + " need to be in input data")
    
    if data_has_predictions:
        model_columns = [c for c in df.columns if c.startswith('y_pred')]
        if not model_columns:
            raise ValueError("If 'data_has_predictions' is True, data must have at least one column whose name starts with 'y_pred'.")
        if model_names is None:
            model_names = ['model_' + str(i) for i in range(len(model_columns))]
        else:
            if len(model_names) != len(model_columns):
                raise ValueError(f"Data contains predictions for {len(model_columns)} models, but only {len(model_names)} model names are given.")

    if not df.isna().sum().any():
        print("Data has no missing values, setting number of imputations M to 0")
        M = 0

    # Generate random seeds from initial random state
    sseq = np.random.SeedSequence(random_state)
    sgen = sseq.spawn(B + B*M + M)
    seeds = np.array([s.generate_state(1)[0] for s in sgen])

    seeds_orig = seeds[:M]  # Seeds for imputing original data
    seeds_boot = seeds[M:]  # Seeds for bootstrap samples and imputing bootstrap samples
    seeds_boot = np.split(seeds_boot, B)
    
    # ==== 1. Compute metrics ====

    # Result container
    data = PerformanceData()

    # ---- 1.1. Compute metrics on original data ----
    print('Computing metrics on original data...')

    # Estimate probability of cancer corresponding to FIT == 10 on original data using complete FIT values
    dfsub = df.loc[~df.fit_val.isna()]
    clf_fit = create_spline_model(dfsub.fit_val.to_numpy(), dfsub.y_true.to_numpy(), knots=[10, 100])
    risk_fit = clf_fit.predict_proba(np.array([10]))[:, 1]
    risk_fit = risk_fit.round(4).item()
    sens_fit = (dfsub.loc[dfsub.fit_val >= 10, 'y_true'].sum() / dfsub.y_true.sum()).item()
    thr_risk = sorted(list(set(thr_risk + [risk_fit])))  # Add risk_fit to risk levels at which models are evaluated
    sens = sorted(list(set(sens + [sens_fit])))  # add sens_fit to sensitivity levels at which models are evaluated
    print('Estimated risk for FIT == 10:', risk_fit)
    print('Estimated sensitivity for FIT >= 10:', sens_fit)

    if plot_fit_model:
        test = dfsub.copy()
        test['pred'] = clf_fit.predict_proba(test[['fit_val']].to_numpy())[:, 1]
        test = test.sort_values(by=['fit_val'])
        plt.plot(test.fit_val, test.pred)
        plt.xticks(np.arange(0, test.fit_val.max(), 100).tolist())
        plt.grid(which='major')
        plt.xlabel('FIT test value (µg/g)')
        plt.ylabel('Predicted probability of CRC')
        plt.axvline(x=10, color='red', linestyle='solid', linewidth=1., alpha=0.5)
        plt.axhline(y=risk_fit, color='red', linestyle='solid', linewidth=1., alpha=0.5)
        msg = 'Probability of cancer for FIT = 10: {:.4} ({:.4f} %)'.format(risk_fit, risk_fit * 100)
        plt.text(0.98, 0.05, msg, 
                 transform=plt.gca().transAxes, 
                 horizontalalignment='right', 
                 verticalalignment='bottom',
                 color='red')
        if save_path is not None:
            plt.savefig(save_path / 'fit-spline-model.png', dpi=100)
        plt.close()
    
    if not fit_spline:  # Should the fitted FIT-only spline model be subsequently evaluated?
        clf_fit = None

    # Get outcome and predictions for each model and the FIT test
    def _outcome_and_prediction_in_long_format(df, b):
        if data_has_predictions:
            df['idx'] = np.arange(df.shape[0])  # Indicates row of the original dataframe in wide format
            df_long = df.melt(id_vars=['y_true', 'idx'], value_vars=['fit_val'] + model_columns, var_name='model_name', value_name='value')
            df_long = df_long.rename(columns={'value': 'y_pred'})
            df_long.model_name = df_long.model_name.replace({i: j for i, j in zip(model_columns, model_names)})
            df_long.model_name = df_long.model_name.replace({'fit_val': 'fit'})
            df_long['m'] = 0  # Indicates that data was not imputed
            df_long['b'] = b  # Indicates the data sample (-1 for original data, 0, 1, 2, ... for bootstrap samples)
        else:
            # Split input data into outcome and predictor variables
            y, x = df[['y_true']], df.drop(labels=['y_true'], axis=1)
            
            # Impute original data M times if it has missing values
            # If there are no missing values, the function just returns original dataframe
            # and adds a single value for the imputed dataset indicator (m = 0)
            x_imp, y_imp, idx_imp = impute_xy(x, y, M, max_iter, seeds_orig, impute_with_y)

            # Get true outcome labels and predicted probabilities for each imputed dataset
            # This applies the models specified in model_names, 
            # applies the FIT-only spline model if clf_fit is not None,
            # and also adds the fit test values as a 'dummy' model.
            df_long = predict(x_imp, y_imp, idx_imp, model_names, repl_ypred_nan, clf_fit, b=-1)
        return df_long

    df_long = _outcome_and_prediction_in_long_format(df, b=-1)

    # Recalibrate predictions
    if recal:
        
        # Estimate recalibration models and apply
        cal = recalibration_models(df_long)
        dfcal = recalibrate_predictions(df_long, cal)
        
        df_long = pd.concat(objs=[df_long, dfcal], axis=0)
        #df_long.groupby(['model_name', 'm', 'b']).size()  # quick check

    # Get max predicted probabilities for each model - for x axis limit of smooth cal curve
    ymax = df_long.groupby('model_name')['y_pred'].max().round(2)

    # Estimate sensitivity of FIT at each threshold in thr_fit
    sens_fit_all = []
    for t in thr_fit:
        s = (df.loc[df.fit_val >= t, 'y_true'].sum() / df.y_true.sum()).item()
        sens_fit_all.append(s)
        print('Estimated sensitivity for FIT >= {}: {}'.format(t, s))

    # For each model, find thresholds that yield the sensitivities corresponding to FIT thresholds
    thr_sens_fit = pd.DataFrame()
    for model_name in df_long.model_name.unique():
        for m in df_long.m.unique():
            dfsub = df_long.loc[(df_long.model_name == model_name) & (df_long.m == m)]
            for t, s in zip(thr_fit, sens_fit_all):
                out = metric_at_single_sens(y_true=dfsub.y_true.to_numpy().squeeze(), 
                                            y_pred=dfsub.y_pred.to_numpy().squeeze(), 
                                            target_sens=s)
                p = pd.DataFrame({'model_name': model_name, 'm': m, 'thr_fit': t, 'sens_fit': s, 'thr_mod': out.thr.item()}, 
                                  index=[0])
                thr_sens_fit = pd.concat(objs=[thr_sens_fit, p], axis=0)
    thr_sens_fit = thr_sens_fit.reset_index(drop=True)
    if save_path is not None:
        thr_sens_fit.to_csv(save_path / 'fit_and_model_thresholds.csv', index=False)

    # Compute metrics for each model, bootstrap sample, and imputed dataset combination
    d = metrics_over_imputations(df_long, ymax, global_only=global_only, interp_step=interp_step, 
                                 thr_risk=thr_risk, sens=sens,
                                 rocpr=True, prob_min=prob_min, format_long=True, raw_rocpr=raw_rocpr,
                                 thr_sens_fit=thr_sens_fit)
    data = _merge_data([data, d])

    # If no bootstrap samples are requested, return metrics computed on original data without CI
    if B == 0:
        return data

    # ---- 1.2. Compute metrics on bootstrapped data ----
    print('\nComputing metrics on bootstrap samples of original data...')
    if B > 0:

        def _boot_iter(i):
            """Runs a single iteration of bootstrap"""
            print('Bootstrap sample {}'.format(i))

            # Get seeds for this iteration: one for taking bootstrap sample, M for imputing data
            seeds_iter = seeds_boot[i]
            
            # Generate a bootstrap sample
            y = df[['y_true']]
            rng = np.random.default_rng(seed=seeds_iter[0])
            if stratified_boot:
                idx0 = np.where(y == 0)[0]
                idx1 = np.where(y == 1)[0]
                idx0_boot = rng.choice(a=idx0, size=len(idx0), replace=True)
                idx1_boot = rng.choice(a=idx1, size=len(idx1), replace=True)
                idx_boot = np.concatenate([idx0_boot, idx1_boot])
            else:
                idx_boot = rng.choice(a=np.arange(len(y)), size=len(y), replace=True)
            df_boot = df.iloc[idx_boot, :]
            
            # Reformat data
            df_long = _outcome_and_prediction_in_long_format(df_boot, b=i)

            # Recalibrate predictions of some models using previously estimated calibration models
            if recal:
                dfcal = recalibrate_predictions(df_long, cal=cal)
                df_long = pd.concat(objs=[df_long, dfcal], axis=0)

            # Compute metrics for each model, bootstrap sample, and imputed dataset combination
            d = metrics_over_imputations(df_long, ymax, global_only=global_only, interp_step=interp_step, 
                                         thr_risk=thr_risk, sens=sens,
                                         rocpr=True, prob_min=prob_min, format_long=True, raw_rocpr=raw_rocpr,
                                         thr_sens_fit=thr_sens_fit)

            # Store roc and pr curve data only for first nroc samples to avoid large objects
            if i > n_noci:
                d.roc = pd.DataFrame()
                d.pr = pd.DataFrame()
            
            return d
        
        def _process_chunk(indices):
            result = []
            for i in indices:
                result_i = _boot_iter(i)
                result.append(result_i)
            return result
        
        if parallel:
            #boot_results = Parallel(n_jobs=-1)(delayed(_boot_iter)(i) for i in range(B))
            chunks = np.array_split(np.arange(B), nchunks)
            boot_results = Parallel(n_jobs=-1)(delayed(_process_chunk)(indices) for indices in chunks)
            boot_results = [item for sublist in boot_results for item in sublist]
        else:
            boot_results = [0] * B
            for i in range(B):
                boot_results[i] = _boot_iter(i)
        
        # Merge metrics computed on bootstrap samples, then merge with original data
        res = _merge_data(boot_results)
        data = _merge_data([data, res])

    if return_boot_samples:
        return data
    
    # ==== 2. Get bootstrap CI for metrics ====

    # Compute bootstrap confidence interval for metrics where it can be computed
    ci_data = boot_ci(data=data, save_path=save_path, method=boot_method)

    # Return metrics on original sample, and on n_noci bootstrap samples
    # for metrics where bootstrap CI cannot be directly computed
    noci_data = boot_noci(data=data, save_path=save_path, nboot=n_noci, seed=random_state)

    # Plot bootstrap distributions
    if plot_boot:
        plot_boot_metrics(data, save_path)

    toc = time.time()
    print('Code ran in {:.2f} minutes.'.format((toc - tic) / 60))
    return ci_data, noci_data


# ~ Helpers: make predictions using existing models ~
def predict(x_imp: pd.DataFrame, y_imp: pd.DataFrame, idx_imp: np.ndarray, model_names: list,
            repl_ypred_nan: bool = False, clf_fit = None, b: int = -1):
    """Helper function to make predictions using models in model_names"""

    # Result container
    df = pd.DataFrame()

    # Make predictions using existing logistic and Cox models (obtained via get_model)
    for mname in model_names:
        model = get_model(mname)
        y_pred = model(x_imp)
        p = pd.DataFrame.from_dict({'y_true': y_imp.y_true.to_numpy(), 
                                    'y_pred': y_pred, 
                                    'm': x_imp['m'].to_numpy(), 
                                    'idx': idx_imp})
        p['model_name'] = mname
        p['b'] = b  # Include sample indicator for compatibility with bootstrap data
        df = pd.concat(objs=[df, p], axis=0)

    # Add predictions made by the spline model?
    if clf_fit is not None:
        y_pred = clf_fit.predict_proba(x_imp.fit_val.to_numpy())[:, 1]
        p = pd.DataFrame.from_dict({'y_true': y_imp.y_true.to_numpy(), 
                                    'y_pred': y_pred, 
                                    'm': x_imp['m'].to_numpy(), 
                                    'idx': idx_imp})
        p['model_name'] = 'fit-spline'
        p['b'] = b  # Include sample indicator for compatibility with bootstrap data
        df = pd.concat(objs=[df, p], axis=0)
    
    # Add FIT test as a dummy model
    if 'fit_val' in x_imp.columns:
        p = pd.DataFrame.from_dict({'y_true': y_imp.y_true.to_numpy(), 
                                    'y_pred': x_imp.fit_val.to_numpy(), 
                                    'm': x_imp['m'].to_numpy(), 
                                    'idx': idx_imp})
        p['model_name'] = 'fit'
        p['b'] = b  # Include sample indicator for compatibility with bootstrap data
        df = pd.concat(objs=[df, p], axis=0)
    
    if repl_ypred_nan:  # For testing only
        df.y_pred = df.y_pred.fillna(0)
    
    return df


# ~ Helpers: Impute data ~
def impute_xy(x: pd.DataFrame, y: pd.DataFrame, M: int, max_iter: int, seeds: np.ndarray, impute_with_y: bool):
    """Imputes a dataset M times if there are missing values.
    Args
        x: DataFrame of predictor variables, shape n x p, where n is number of patients and p is number of variables
        y: DataFrame of the outcome variable, shape n x 1
        M: number of imputations
        max_iter: maximum number of iterations for MICE
        seeds: random states used for each multiple imputation round, must be of the same length as M
        impute_with_y: if True, the outcome variable y is used to impute missing values in predictor variables,
                  (but the outcome variable itself is not imputed and is assumed to be complete).
    Returns:
        x_imp: DataFrame containing imputed predictor variables, 
               where m indicates the number of imputation (e.g. 0 - first imputed dataset, 1 - second imputed dataset)
        y_imp: pd.DataFrame containing the outcome variable corresponding to each row in x_imp
        idx_imp: np.ndarray containing the index of each original sample. 
                 For example, index 0 returns M rows from x_imp that correspond to the first sample.
    
        If input data x does not contain missing values, the outputs x_imp and y_imp are the same as input data x and y,
        and y_imp contains the row numbers of the input data.
    """
    has_mis = x.isna().values.any()
    if has_mis:
        print('... Imputing missing values on original data')

        # Use outcome variable (y) in imputation if requested
        if impute_with_y:
            df = pd.concat(objs=[x, y], axis=1) 
            df_imp = _impute(df, M=M, max_iter=max_iter, seeds=seeds)
            x_imp = df_imp.loc[:, x.columns.tolist() + ['m']]
            y_imp = df_imp.loc[:, y.columns]
        else:  # Otherwise use predictor variables only
            x_imp = _impute(x, M=M, max_iter=max_iter, seeds=seeds)
            y_imp = pd.concat(objs=[y for i in range(M)], axis=0)
        
        # Also get indices of imputed samples
        idx_imp = np.concatenate([np.arange(x.shape[0]) for i in range(M)])
    else:
        x_imp = x
        x_imp['m'] = 0
        y_imp = y
        idx_imp = np.arange(x.shape[0])
    
    return x_imp, y_imp, idx_imp


def _impute(x: pd.DataFrame, M: int, max_iter: int = 10, seeds: int = None):
    """Impute a dataset M times using a Bayesian normal linear model
    Args
        x : input DataFrame to be imputed
        M : number of imputations
        max_iter : maximum number of iterations for generating each imputed dataset
        seed_imputer : initial random state for imputations
    """
    if seeds is not None and len(seeds) != M:
        raise ValueError("If seeds are supplied, there must be M of them")
    
    # Impute M times 
    x_imp = pd.DataFrame()
    for i in range(M):
        print('... Generating imputed dataset', i)
        if seeds is not None:
            imputer_state = seeds[i]
        else:
            imputer_state = None

        impute_estimator = BayesianRidge()
        imputer = IterativeImputer(random_state=imputer_state, estimator=impute_estimator,
                                   max_iter=max_iter, sample_posterior=True, min_value=0)
        imputer.fit(x)
        xm = imputer.transform(x)
        xm = pd.DataFrame(xm, columns=x.columns)
        xm['m'] = i
        x_imp = pd.concat(objs=[x_imp, xm], axis=0)
    
    return x_imp


# ~ Helpers: compute metrics over imputed datasets ~
def metrics_over_imputations(df: pd.DataFrame, ymax: pd.DataFrame, 
                             thr_risk: list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2], 
                             thr_fit: list = [2, 10, 100],
                             sens: list = [0.8, 0.85, 0.9, 0.95, 0.99], 
                             global_only: bool = False,
                             interp_step: float = 0.01, 
                             prob_min: float = 0.2, 
                             rocpr: bool = True, 
                             format_long: bool = True,
                             raw_rocpr: bool = False, 
                             thr_sens_fit: pd.DataFrame = None,
                             dca: bool = True):
    """For each model in df, compute metrics on each imputed dataset,
    then take the average value of each metric over imputations.
    
    Args
        df: DataFrame with the following columns:
            y_true: true 0-1 outcome labels, e.g. 0: no cancer, 1: cancer
            y_pred: predicted probabilities of cancer according to one or more models
            model_name: name of the model
            m: indicates the multiple imputation round
            b: indicates the bootstrap sample
            Note that df must also contain data for the fit test, under model_name 'fit',
            and in this case y_pred does not contain probabilities but FIT values.
        
        ymax: DataFrame containing the maximum predicted probability for each model,
              with columns
             
    Assumes that df contains data for a single bootstrap sample, indexed by b in [0, 1, ...]
    and for multiple imputations, indexed by m in [0, 1, ...].
    df must also contain FIT test values under model_name == 'fit', 
    needed for computing calibration curves restricted to certain ranges of FIT.
    """

    # Input check
    columns = ['y_true', 'y_pred', 'model_name', 'b', 'm']
    if not np.all([c in df.columns for c in columns]):
        raise ValueError("df must contain all columns in " + str(columns))
    if not np.any(df.model_name == 'fit'):
        raise ValueError("df must contain data for FIT test, under model_name = 'fit'")

    # Result container, boostrap sample indicator, FIT test values
    # In predict.py, FIT test values are included once for original data and each bootstrap sample
    data = PerformanceData()
    b = df.b.iloc[0]
    fit = df.loc[df.model_name == 'fit', 'y_pred'].to_numpy().squeeze()

    # Loop over models, then over imputed datasets
    models = df.model_name.unique()
    for model_name in models:
        if model_name == 'fit':
            cal = False  # If model is 'fit' (FIT test values), calibration metrics cannot be computed
        else:
            cal = True

        # Compute metrics for this model on all imputed datasets
        model_data = PerformanceData()
        imputations = df.loc[df.model_name == model_name, 'm'].unique()
        for m in imputations:

            # True labels and predicted probabilities for current model on current imputed data
            dfsub = df.loc[(df.model_name == model_name) & (df.m == m)]
            y_true = dfsub.y_true.to_numpy().squeeze()
            y_pred = dfsub.y_pred.to_numpy().squeeze()

            # FIT test values
            dffit = df.loc[(df.model_name == 'fit') & (df.m == m)]
            fit = dffit.y_pred.to_numpy().squeeze()

            # Max predicted probability for each model
            ym = ymax[model_name].item()

            # Add prediction threshold yielding sensitivity of FIT >= 10
            # These are specific to the current model and imputed dataset, and taken from thr_sens_fit DataFrame
            if thr_sens_fit is not None and model_name not in ['fit', 'fit-spline']:
                dfthr = thr_sens_fit.loc[(thr_sens_fit.model_name == model_name) & (thr_sens_fit.m == m)]
                thr_fit_use = dfthr.thr_fit.tolist()
                thr_mod_use = dfthr.thr_mod.tolist()

                #thr_risk = thr_risk + thr_mod_use  # Adds a bit too many values, esp if multiple FIT thrs and models.
                #thr_risk = list(set(thr_risk))
            else:
                thr_fit_use = thr_fit
                thr_mod_use = None

            # Compute metrics
            metrics = all_metrics(y_true, y_pred, fit, thr_fit=thr_fit_use, thr_risk=thr_risk, sens=sens,
                                  interp_step=interp_step, ymax_bin=[prob_min, ym], ymax_lowess=[prob_min, ym], 
                                  global_only=global_only, cal=cal, rocpr=rocpr,
                                  wilson_ci=False, format_long=format_long, 
                                  raw_rocpr=raw_rocpr, thr_mod=thr_mod_use, dca=dca)

            metrics = _add_groupings(metrics, model_name=model_name, b=b, m=m)

            # Merge
            model_data = _merge_data([model_data, metrics])
        
        # Take the average value of metrics (for which average can be computed) over imputed datasets for this model, and store
        if len(imputations) > 1:
            model_data = _average_over_imputations(model_data)
        data = _merge_data([data, model_data])
    
    # For some metrics, performance of the FIT test is computed at each call to all_metrics
    # Remove duplicate entries and tidy it a bit
    remove_duplicate_rows = True
    if remove_duplicate_rows:
        df = data.dc
        if df.shape[0] > 0:
            mods = ['all', 'none', 'fit10']
            df0 = df.loc[df.model.isin(mods)].drop_duplicates(subset=['threshold', 'model', 'metric_name'])
            for m in mods:
                df0.loc[df0.model == m, 'model_name'] = m
            df1 = df.loc[~df.model.isin(mods)]
            df = pd.concat(objs=[df0, df1], axis=0)
            data.dc = df

    return data


def _add_groupings(data: PerformanceData, model_name: str, b: int, m: int):
    """Add model name (model_name), bootstrap sample indicator (b) 
    and imputed dataset indicator (m) to each table in PerformanceData class"""

    fields = [f for f, __ in data.__dataclass_fields__.items()]
    for f in fields:
        d = getattr(data, f)
        d = d.copy()
        if d.shape[0] > 0:
            d['model_name'] = model_name
            d['b'] = b
            d['m'] = m
            setattr(data, f, d)

    return data


def _merge_data(inp: list):
    """Merge instances of PerformanceData by field 
    - e.g. to merge global performance metrics over imputed datasets"""
    merged_data = PerformanceData()

    # Identify fields - all dataclasses in inp must have same fields
    fields = [f for f, __ in inp[0].__dataclass_fields__.items()]
    
    # Merge by field
    for f in fields:
        d = pd.concat(objs=[getattr(i, f) for i in inp], axis=0)
        setattr(merged_data, f, d)

    return merged_data


def _average_over_imputations(data: PerformanceData):
    """Compute average statistic over imputed datasets in each table within the PerformanceData class
    
    Assumes the data is in long format, so 'metric_value' column contains the values to average
    and the 'm' column contains the imputed dataset indicator to take the average over
    and all other columns can be kept as index

    NB average over imputations is not meaningful for some data items, such as non-interpolated ROC curve data.
    """

    # Data items not meant to be averaged as averaging is not meaningful
    fields_exclude = ['roc', 'pr', 'cal_bin']

    # Take average
    fields = [f for f, __ in data.__dataclass_fields__.items()]
    fields = [f for f in fields if f not in fields_exclude]
    for f in fields:
        d = getattr(data, f)
        if d.shape[0] > 0: 
            valuecol = 'metric_value'
            groupcols = [c for c in d.columns if c not in ['m', 'metric_value']]
            d = d.groupby(groupcols)[valuecol].mean().reset_index()  # This could fail for columns with object dtype (e.g. when one row np array, another scalar), not sure why
            setattr(data, f, d)

    return data


# ~ Helpers: Compute bootstrap confidence intervals ~
def boot_ci(data: PerformanceData, save_path: Path = None, method='percentile'):
    """Compute boostrap confidence intervals"""
    print('\nComputing bootsrap confidence intervals...')

    # Data items not meant to be averaged as averaging is not meaningful
    fields_exclude = ['roc', 'pr', 'cal_bin']

    # Fields to compute CI for
    fields = [f for f, __ in data.__dataclass_fields__.items()]
    fields = [f for f in fields if f not in fields_exclude]

    # Get bootstrap CI
    ci = PerformanceData()
    for f in fields:
        print('\nComputing bootstrap CI for table {}'.format(f))
        d = getattr(data, f)
        if d.shape[0] > 0:
            value_col = 'metric_value'
            group_cols = [c for c in d.columns if c not in ['b', 'metric_value']]
            print('Columns: {}'.format(d.columns.tolist()))
            print('Group columns: {}'.format(group_cols))
            print('Value column: {}'.format(value_col))
            d = _add_ci(d, group_cols, value_col, method)
            setattr(ci, f, d)
    
    if save_path is not None:
        print('\nSaving data to disk...')
        ci.disc.to_csv(save_path / DISC_CI, index=False)
        ci.cal.to_csv(save_path / CAL_CI, index=False)
        
        ci.thr_risk.to_csv(save_path / RISK_CI, index=False)
        ci.thr_sens.to_csv(save_path / SENS_CI, index=False)
        ci.thr_sens_fit.to_csv(save_path / SENS_FIT_CI, index=False)
        ci.thr_fit_mod.to_csv(save_path / SENS_FIT_2_CI, index=False)
        
        ci.roc_int.to_csv(save_path / ROC_CI, index=False)
        ci.pr_int.to_csv(save_path / PR_CI, index=False)
        ci.pr_gain.to_csv(save_path / PR_GAIN_CI, index=False)
        ci.cal_smooth.to_csv(save_path / CAL_SMOOTH_CI, index=False)

        ci.dc.to_csv(save_path / DC_CI, index=False)
        print('Save complete.')
    
    return ci


def boot_noci(data: PerformanceData, save_path: Path = None, nboot: int = 20, seed: int = 42):
    """Select a smaller number of bootstrap samples for data items 
    where it is not possible to compute a bootstrap CI"""

    print("\nSelecting nboot samples for data items where bootstrap CI cannot be computed...")

    @dataclass
    class DataNoCI:
        roc: pd.DataFrame = pd.DataFrame()
        pr: pd.DataFrame = pd.DataFrame()
        cal_bin: pd.DataFrame = pd.DataFrame()

    rng = np.random.default_rng(seed=seed)
    data_noci = DataNoCI()

    for f in ['roc', 'pr', 'cal_bin']:

        d = getattr(data, f)
        if d.shape[0] == 0:
            continue

        # Separate original data and bootstrap samples
        d0 = d.loc[d.b == -1]
        dboot = d.loc[d.b != -1]

        # Select nboot bootstrap samples, using first imputed dataset
        # For thr_obs data item, do not include bootstrap samples
        if dboot.shape[0] > 0:
            idx = dboot.b.unique()
            n = np.min([len(idx), nboot])
            b = rng.choice(idx, size=n, replace=False)
            dsel = dboot.loc[dboot.b.isin(b) & (dboot.m == 0)]
        else:
            dsel = pd.DataFrame()

        # Merge and store
        dsub = pd.concat(objs=[d0, dsel], axis=0)
        setattr(data_noci, f, dsub)
        
    if save_path is not None:
        data_noci.roc.to_csv(save_path / ROC_NOCI, index=False)
        data_noci.pr.to_csv(save_path / PR_NOCI, index=False)
        data_noci.cal_bin.to_csv(save_path / CAL_BIN_NOCI, index=False)

    return data_noci


def q025(x):
    return np.quantile(x, q=0.025)


def q975(x):
    return np.quantile(x, q=0.975)


def _add_ci(df: pd.DataFrame, group_cols: list, value_col: list, method: str = 'percentile', clip: tuple = None):
    """Get bootsrap CI"""

    # Separate original and boostrap samples
    df0 = df.loc[df.b == -1]
    dfb = df.loc[df.b != -1]

    if dfb.shape[0] == 0:
        warnings.warn('Table contains no boostrap samples, returning original table.')
        return df0
    
    # If value_col is list, pivot
    if isinstance(value_col, list):
        df0 = pd.melt(df0, id_vars=group_cols, value_vars=value_col, var_name='metric_name', value_name='metric_value')
        dfb = pd.melt(dfb, id_vars=group_cols, value_vars=value_col, var_name='metric_name', value_name='metric_value')
        value_col = 'metric_value'
        group_cols += ['metric_name']

    # Get 2.5th and 97.5th percentiles
    q = dfb.groupby(group_cols)[value_col].agg([q025, q975])
    #q.columns = q.columns.droplevel(0)
    q = q.reset_index()
    ci = df0.merge(q, how='left')
    if 'b' in ci.columns:
        ci = ci.drop(labels=['b'], axis=1)

    # Get CI
    if method == 'basic':
        ci['ci_low'] = ci[value_col] * 2 - ci.q975
        ci['ci_high'] = ci[value_col] * 2 - ci.q025

        if clip is not None:
            ci.ci_low = ci.ci_low.clip(clip[0], clip[1])
            ci.ci_high = ci.ci_high.clip(clip[0], clip[1])

    elif method == 'percentile':
        ci['ci_low'] = ci.q025
        ci['ci_high'] = ci.q975
    
    return ci


# ~ Helpers: Recalibrate models ~
def _logit(x):
    return np.log(x/(1 - x))


def recalibration_models(df: pd.DataFrame, models: list = None):
    """Estimate models that recalibrate originally fitted models"""

    # Models to be recalibrated
    if models is None:
        models = df.model_name.unique().tolist()
    else:
        models = [m for m in models if m in df.model_name.unique()]
    models = [m for m in models if m not in ['fit', 'fit-spline']]

    # Result container
    cal = {'platt': {m: [] for m in models}}
    
    imputations = df.m.unique()

    for model_name in models:
        for m in imputations:

            # Dataset containing predictions for current model in imputed dataset m
            dfsub = df.loc[(df.model_name == model_name) & (df.m == m)]
            y_true = dfsub.y_true.to_numpy()
            y_pred = dfsub.y_pred.to_numpy().reshape(-1, 1)

            # Parameters of Platt scaling
            calibrator = LogisticRegression(penalty=None)
            if np.any(np.isin(y_pred, [0, 1])):
                warnings.warn('Some predicted probabilities are exactly 0 or 1. Replacing with 0.00001 or 0.99999 before logist recalibration')
            y_pred[y_pred == 0] = 0.00001
            y_pred[y_pred == 1] = 0.99999
            calibrator.fit(_logit(y_pred), y_true)
            cal['platt'][model_name].append(calibrator)
    
    return cal


def recalibrate_predictions(df: pd.DataFrame, cal: dict):
    # Recalibrate original data
    print('Recalibrating predictions...')

    models = list(cal['platt'].keys())
    imputations = df.m.unique()

    res = pd.DataFrame()
    for model_name in models:
        for m in imputations:
            
            # Dataset containing predictions for current model in imputed dataset m
            dfsub = df.loc[(df.model_name == model_name) & (df.m == m)].copy()
            y_pred = dfsub.y_pred.to_numpy().reshape(-1, 1)

            # Apply logistic regression to recalibrate predictions
            calibrators = cal['platt'][model_name]
            y_pred[y_pred == 0] = 0.00001
            y_pred[y_pred == 1] = 0.99999
            y_cal = []
            for c in calibrators:
                tmp = c.predict_proba(_logit(y_pred))[:, 1]
                y_cal.append(tmp.reshape(-1, 1))
            y_cal = np.hstack(y_cal).mean(axis=1).reshape(-1)

            df_cal = dfsub.copy()
            df_cal.y_pred = y_cal
            df_cal.model_name = df_cal.model_name + '-platt'
            res = pd.concat(objs=[res, df_cal], axis=0)

    return res


# ~ Helpers: plot Bootstrap distributions ~
def _plot_boot(df, save_path, out_name):

    # Columns that contain model name, metric name and metric value
    model_name_col = 'model_name'
    metric_name_col = 'metric_name'
    metric_value_col = 'metric_value'
    
    # Index columns
    #  In addition to model name, performance metric tables may contain other columns 
    #  that act as grouping variables, such as level of sensitivity or FIT threshold
    #  we want to plot bootstrap distributions seperately for these groups.
    non_index_cols = ['metric_name', 'metric_value', 'model_name', 'b', 'm', 'model']
    index_cols = [c for c in df.columns if c not in non_index_cols]
    if index_cols:
        groups = df[index_cols + [model_name_col]].drop_duplicates().reset_index(drop=True)
        if len(groups) > 10:
            #groups = groups.groupby(model_name_col).sample(1, random_state=42).reset_index(drop=True)
            groups = groups.sample(10, random_state=42).reset_index(drop=True)
    else:
        index_cols = None
        groups = df[[model_name_col]].drop_duplicates().reset_index(drop=True)
    
    # Models to include
    models = df[model_name_col].unique()
    models_excl = ['all', 'none', 'fit10']  # Exclude 'all', 'none', 'fit10' from DC curves
    models = [m for m in models if m not in models_excl]
    groups = groups.loc[groups.model_name.isin(models)]

    # Metrics to include
    metrics = df[metric_name_col].unique()
    metrics_incl = ['auroc', 'ap', 
                    'oe_ratio', 'log_intercept', 'log_slope',
                    'sens', 'spec', 'ppv', 'npv', 'pp',
                    'sens_fit', 'spec_fit', 'ppv_fit', 'npv_fit', 'pp_fit', 'thr_fit',
                    'sens_mod', 'spec_mod', 'ppv_mod', 'npv_mod', 'pp_mod', 'thr_mod',
                    'proportion_reduction_tests', 'delta_sens', 'delta_ppv',
                    'tpr', 'precision', 'precision_fit',
                    'prob_true',
                    'net_benefit', 'net_intervention_avoided'
                    ]
    metrics = [m for m in metrics if m in metrics_incl]

    # Figure layout
    nrow = groups.shape[0]
    ncol = len(metrics)
    fig_width = ncol * 3
    fig_height = nrow * 2
    figsize = (fig_width, fig_height)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(nrows=nrow, ncols=1)

    for i, row in groups.iterrows():
        
        # Model name
        model_name = row.model_name

        # Title for current row
        subfig = subfigs[i]
        if index_cols:
            title = ""
            for c in index_cols:
                title += c + ':' + str(np.round(row[c], 3)) + ', '
            title += 'model: ' + model_name
        else:
            title = 'Model: ' + model_name
        subfig.suptitle(title, x=0.0, ha='left', fontsize=10) #, fontweight='bold')

        # Plot metrics into each column of this row
        dfrow = pd.DataFrame(row).transpose().merge(df, how='inner')

        ax = subfig.subplots(nrows=1, ncols=len(metrics))     
        for j, metric_name in enumerate(metrics):
            
            # Bootstrap distribution
            dfsub = dfrow.loc[(dfrow.metric_name == metric_name) & (dfrow.b != -1), metric_value_col]
            sns.kdeplot(dfsub, ax=ax[j], label=model_name)

            # Estimate based on original data
            dforig = dfrow.loc[(dfrow.metric_name == metric_name) & (dfrow.b == -1), metric_value_col]
            if dforig.shape[0] > 0:
                value_orig = dforig.iloc[0]
                ax[j].axvline(x=value_orig, color='gray') 

            # Adjust
            ax[j].set_title(metric_name, fontsize=8)
            ax[j].set_xlabel(xlabel='metric_value', fontsize=8)
            ax[j].set_ylabel(ylabel='density', fontsize=8)
            if dfsub.std() < 1e-5:   # If there is almost no variation in the metric over samples do not plot.
                ax[j].set_visible(False)
                
    plt.subplots_adjust(top=0.75, hspace=0.75, wspace=0.5)
    plt.savefig(save_path / out_name, dpi=75, facecolor='white', bbox_inches='tight')
    plt.close()


def plot_boot_metrics(data_ci, save_path):
    """Plot bootstrap distributions"""

    # Exclude some tables - mainly to save time and reduce computation
    # Some tables, like cal_bin and cal_smooth may also have more than 1 index col
    # and are not incorporated atm.
    fields_excl = ['cal_bin', 'pr', 'roc']
    fields = [f for f, __ in data_ci.__dataclass_fields__.items()]
    fields = [f for f in fields if f not in fields_excl]
    for f in fields:
        print('\n...plotting bootstrap distributions for', f)
        d = getattr(data_ci, f)
        d = d.copy()
        if d.shape[0] > 0:
            out_name = 'plot_boot_data-' + f + '.png'
            try:
                _plot_boot(d, save_path, out_name)
            except Exception as ex:
                print("_plot_boot did not complete for table ", f, ", exception: ", ex)
