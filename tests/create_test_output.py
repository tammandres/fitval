from fitval.boot import boot_metrics
from constants import PROJECT_ROOT
from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca, plot_cal_bin
from fitval.reformat import reformat_disc_cal

# Output path
test_path = PROJECT_ROOT / 'tests' / 'test_boot'
test_path.mkdir(exist_ok=True, parents=True)

# Paths to data created by create_dummy_data.py (ran for 10.45 minutes)
data_path = PROJECT_ROOT / 'tests' / 'test_data' / 'pred.csv'
data_ci, data_noci = boot_metrics(data_path=data_path, save_path=test_path, data_has_predictions=True, 
                                  model_names=['logistic-full', 'logistic-fast'], B=1000, 
                                  parallel=True, nchunks=10, plot_boot=True, plot_fit_model=True,
                                  thr_mod_add=[0.03, 0.006])

# Plot
plot_cal_bin(test_path, models_incl=['logistic-full', 'logistic-fast', 'fit-spline'])
plot_cal_smooth(test_path, models_incl=['logistic-full', 'logistic-fast', 'fit-spline'])
for ci in [True, False]:
    plot_rocpr_interp(test_path, ci=ci, models_incl=['logistic-full', 'logistic-fast', 'fit'])
    plot_dca(test_path, ci=ci, models_incl=['logistic-full-platt', 'logistic-fast-platt', 'fit-spline'])

# Reformat
reformat_disc_cal(data_path=test_path, save_path=test_path)
