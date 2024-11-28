from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca, plot_cal_bin
from constants import PROJECT_ROOT


# Output path
test_path = PROJECT_ROOT / 'tests' / 'test_boot_new'
test_path.mkdir(exist_ok=True, parents=True)


def test_plot_cal_smooth():
    plot_cal_smooth(test_path)


def test_plot_rocpr_interp():
    plot_rocpr_interp(test_path)


def test_plot_dca():
    plot_dca(test_path)


def test_plot_cal_bin():
    plot_cal_bin(test_path)
