from fitval.boot import boot_metrics
from constants import PROJECT_ROOT
from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca

# Output path
test_path = PROJECT_ROOT / 'tests' / 'test_boot'
test_path.mkdir(exist_ok=True, parents=True)

# Paths to data created by create_dummy_data.py
path_pred = test_path / 'pred.csv'
data_ci, data_noci = boot_metrics(data_path=path_pred, save_path=test_path, data_has_predictions=True, 
                                  model_names=['logistic-full', 'logistic-fast'], B=1000, 
                                  parallel=True, nchunks=10, plot_boot=False, plot_fit_model=False)

# Plot
plot_cal_smooth(test_path)
for ci in [True, False]:
    plot_rocpr_interp(test_path, ci=ci)
    plot_dca(test_path, ci=ci)
    