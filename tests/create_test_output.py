from fitval.boot import boot_metrics, boot_reduction_in_referrals
from constants import PROJECT_ROOT
from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca, plot_cal_bin, plot_reduction
from fitval.reformat import reformat_disc_cal

# Data path (contains model predictions)
data_path = PROJECT_ROOT / 'tests' / 'test_data' / 'pred.csv'

# Apply boot metrics
test_path = PROJECT_ROOT / 'tests' / 'test_boot_metrics'
test_path.mkdir(exist_ok=True, parents=True)
data_ci, data_noci = boot_metrics(data_path=data_path, save_path=test_path, data_has_predictions=True, 
                                  model_names=['logistic-full', 'logistic-fast'], B=1000, 
                                  parallel=True, nchunks=10, plot_boot=True, plot_fit_model=True,
                                  thr_mod_add=[0.03, 0.006])

# Plot results of boot metrics
plot_cal_bin(test_path, models_incl=['logistic-full', 'logistic-fast', 'fit-spline'])
plot_cal_smooth(test_path, models_incl=['logistic-full', 'logistic-fast', 'fit-spline'])
for ci in [True, False]:
    plot_rocpr_interp(test_path, ci=ci, models_incl=['logistic-full', 'logistic-fast', 'fit'])
    plot_dca(test_path, ci=ci, models_incl=['logistic-full-platt', 'logistic-fast-platt', 'fit-spline'])
plot_reduction(test_path)

# Reformat results of boot metrics
reformat_disc_cal(data_path=test_path, save_path=test_path)

# Apply boot_reduction_in_referrals
test_path = PROJECT_ROOT / 'tests' / 'test_boot_reduction'
test_path.mkdir(exist_ok=True, parents=True)
metrics, metrics_reformat = boot_reduction_in_referrals(data_path=data_path, save_path=test_path, 
                                                        model_names=['logistic-full', 'logistic-fast'], B=100,
                                                        thr_fit = [2, 10], thr_mod_add=[0.03], return_boot_samples=False)
plot_reduction(test_path)
