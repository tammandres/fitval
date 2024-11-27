from fitval.boot import boot_metrics
from constants import PROJECT_ROOT


# Output path
test_path = PROJECT_ROOT / 'tests' / 'test_boot_new'
test_path.mkdir(exist_ok=True, parents=True)

# Paths to data created by create_dummy_data.py
path_pred = test_path / 'pred.csv'
path_xy = test_path / 'xy.csv'
path_xy_mis = test_path / 'xy_mis.csv'


# Run with some of the different settings
def test_boot_metrics_when_data_has_predictions():
    data_ci, data_noci = boot_metrics(data_path=path_pred, save_path=test_path, model_names=['logistic-full', 'logistic-fast'], 
                                      data_has_predictions=True, B=5, plot_boot=False)


def test_boot_metrics():
    data_ci, data_noci = boot_metrics(data_path=path_xy, save_path=test_path, model_names=['logistic-full', 'logistic-fast'], 
                                      data_has_predictions=False, B=5, plot_boot=False)


def test_boot_metrics_mis():
    data_ci, data_noci = boot_metrics(data_path=path_xy_mis, save_path=test_path, model_names=['logistic-full', 'logistic-fast'], 
                                      data_has_predictions=False, B=5, M=3, plot_boot=False)


def test_boot_metrics_when_data_has_predictions_with_plot_boot():
    data_ci, data_noci = boot_metrics(data_path=path_pred, save_path=test_path, data_has_predictions=True, model_names=None, B=5, 
                                      plot_boot=True)


def test_boot_metrics_parallel():
    data_ci, data_noci = boot_metrics(data_path=path_pred, save_path=test_path, data_has_predictions=True, model_names=None, B=20, 
                                      parallel=True, nchunks=4, plot_boot=False, plot_fit_model=False)
