from fitval.boot import boot_metrics, boot_reduction_in_referrals
from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca, plot_cal_bin
from constants import PROJECT_ROOT


# Paths
#  data_path: path to a csv file that contains the following columns:           
#             `y_true` : indicator variable for colorectal cancer (0: no, 1: yes), 
#             `fit_val` : numerical FIT test values (1.5, 10, 20, etc), 
#             `y_pred` : predicted probability of CRC according to a model, must be in the [0, 1] interval (0.03, 0.05, 0.25 etc). 
#                        Multiple models can also be included, in which case the columns must be named as y_pred1, y_pred2, ... etc.
#             An example table with simple synthetic data is located in `./data/predictions.csv`
#  save_path: path where to save the results
data_path = PROJECT_ROOT / 'data' / 'predictions.csv'
save_path = PROJECT_ROOT / 'results'
save_path.mkdir(exist_ok=True, parents=True)


# Compute all performance metrics over 100 bootstrap samples
#  The main arguments are: 
#    data_path: path to predictions.csv
#    save_path: path where to save results
#    model_names: optional. If submitted, must represent the names of the models that correspond to the predicted probabilities
#                 of cancer in input dataframe. In this example, 'fit-age-sex-bloods' corresponds to y_pred1 and 'fit-age-sex' to y_pred2.
#                 If supplied, the model_name column in performance metrics table will contain these names.
#    data_has_predictions: True by default, specifies that input data contains predicted probabilities for each person.
#                          If False, the get_model function in fitval/models.py must return 
#                          the corresponding prediction model equation in response to its name.
#                          Useful when data needs to be imputed.
#    B: number of bootstrap samples to take. Set to 100 in this example to make it run faster.
#    parallel: if True, the bootstrap samples are processed in parallel in nchunks number of chunks.
#              Maximum number of available cores are used. If less need to be used, the njobs variable in boot.py
#              can be set to the desired number of parallel processes.
#              NB - if parallel is True, the code can sometimes fail with the 'main thread is not in main loop' error
#              if it is run again in the same python environment. I am not sure what causes it, but it could be 
#              related to plots. However, if the python environment is restarted and the code is run fresh
#              it should work, and has worked in this example.
#    nchunks: number of chunks to divide the dataset into, to be processed in parallel (applied only if parallel=True)
#    plot_boot: if True, plots the bootstrap distributions for each performance metric.
#    recal: if True, logistic recalibration is applied to the prediction models, and recalibrated models are validated
#           alongside the original models (their names will have the -platt suffix)
data_ci, data_noci = boot_metrics(data_path=data_path, save_path=save_path, model_names=['fit-age-sex-bloods', 'fit-age-sex'], 
                                  data_has_predictions=True, B=100, parallel=False, nchunks=15, plot_boot=False, recal=False)

# Plot performance metrics 
plot_cal_smooth(save_path)  # plot smooth calibration curves based on LOWESS smoothing
plot_rocpr_interp(save_path)  # plot interpolated ROC and precision-recall curves
plot_dca(save_path)  # plot decision curves
plot_cal_bin(save_path)  # plot calibration curves based on dividing data into bins


# If the aim is to only compute reduction in referrals relative to the FIT test, 
# this can be achieved faster using the boot_reduction_in_referrals function 
# Here, thr_fit represents the thresholds of the FIT test against which the models are to be compared
# The 'metrics' object contains key metrics for the FIT test at threshold 10 and for the model at threshold thr_mod,
#   where thr_mod is chosen such that it captures the same number of cancers as FIT >= 10.
#   The 'metric_name' column specifies what each metric represents:
#      sens_fit: sensitivity of FIT >= 10, 
#      ppv_fit: ppv of FIT >= 10,
#      sens_mod: sensitivity of model (same as sens_fit due to how threshold was chosen), 
#      ppv_mod: ppv of model at threshold that captures the same number of cancers as FIT >= 10
#      proportion_reduction_tests: proportion reduction in the number of positive tests compared to FIT >= 10,
#                                  approximates the reduction in referrals obtained when using the model over FIT
#      delta_sens: difference in sensitivities between model and FIT (0 in this case due to how the model threshold was chosen)
#      delta_tp: difference in the number of true positives between model and FIT (0 in this case due to how the model threshold was chosen)
#   The 'metric_value' gives the point estimate of the metric, and 'ci_low' and 'ci_high' give 95% bootstrap percentile confidence intervals
# The 'metrics_reformat' object contains the same metrics in a wide format and with more easily understandable column names
metrics, metrics_reformat = boot_reduction_in_referrals(data_path, save_path, model_names=['fit-age-sex-bloods', 'fit-age-sex'],
                                                        thr_fit = [10], B=100, plot_boot=True)


# The reduction in referrals can also be computed more manually using the metric_at_fit_sens and metric_at_fit_and_mod_threshold functions
import pandas as pd
import numpy as np
from tqdm import tqdm
from fitval.metrics import metric_at_fit_sens, metric_at_fit_and_mod_threshold

df = pd.read_csv(data_path)  # read data that contains cancer indicator, FIT test values and predicted probabilities 
B = 100  # number of bootstrap samples
seed = 42  # random seed for reproducibility
rng = np.random.default_rng(seed=seed)

y_true, fit_val, y_pred = df.y_true.to_numpy(), df.fit_val.to_numpy(), df.y_pred1.to_numpy()  # using y_pred1 atm - predictions of the first model

# Compute risk score threshold that captures the same number of cancers as FIT >= 10
m_tmp = metric_at_fit_sens(y_true, y_pred, fit_val, thr_fit = [10])
thr_mod = m_tmp.thr_mod.item()

# Compute performance metrics at this threshold 
#  Note that these performance metrics were computed by metric_at_fit_sens too, but I am computing it again by explicitly setting 
#  the model threshold. This ensures that there is no interpolation of performance metrics 
#  if it is not possible to find a threshold that exactly captures the same number of cancers as FIT.
metrics = metric_at_fit_and_mod_threshold(y_true, y_pred, fit_val, thr_mod=[thr_mod], thr_fit=[10])
metrics = pd.melt(metrics, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')

# Bootstrap
m_boot = pd.DataFrame()
for b in tqdm(range(B)):
    idx_boot = rng.choice(a=np.arange(len(y_true)), size=len(y_true), replace=True)
    y_boot, pred_boot, fit_boot = y_true[idx_boot], y_pred[idx_boot], fit_val[idx_boot]

    mb = metric_at_fit_and_mod_threshold(y_boot, pred_boot, fit_boot, thr_mod=[thr_mod], thr_fit=[10])
    mb = pd.melt(mb, id_vars=['thr_mod', 'thr_fit'], value_name='metric_value', var_name='metric_name')
    m_boot = pd.concat(objs=[m_boot, mb], axis=0)

q = m_boot.groupby(['thr_mod', 'thr_fit', 'metric_name']).metric_value.agg([lambda x: x.quantile(0.025), 
                                                                            lambda x: x.quantile(0.975)])
q.columns = ['ci_low', 'ci_high']
q = q.reset_index()
metrics = metrics.merge(q, how='left')
