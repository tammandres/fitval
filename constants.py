from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# Output file names for metrics computed in boot.py 
#   CI: bootstrap CI is computed
#   NOCI: no bootstrap CI is computed
DISC_CI = 'discrimination.csv'  # Overall discrimination metrics (PerformanceData.disc)
CAL_CI = 'calibration.csv'  # OVerall calibration metrics (PerformanceData.cal)

CAL_SMOOTH_CI = 'calibration_curve_smooth.csv'  # Smooth calibration curve data (PerformanceData.cal_smooth)
CAL_BIN_NOCI = 'calibration_curve_binned.csv'  # Binned calibration curve data (PerformanceData.cal_bin)

RISK_CI = 'metrics_at_risk.csv'  # Diagnostic metrics at predefined model thresholds (PerformanceData.thr_risk)
SENS_CI = 'metrics_at_sens.csv'  # Diagnostic metrics at predefined sensitivity levels (PerformanceData.thr_sens)

SENS_FIT_CI = 'metrics_at_sens_fit.csv'  # Diagnostic metrics computed at sensitivity of each FIT threshold (PerformanceData.thr_sens_fit)
SENS_FIT_2_CI = 'metrics_at_fit_and_mod_thresholds.csv'  #  Metrics at FIT thresholds and corresponding model thresholds (PerformanceData.thr_fit_mod)

ROC_CI = 'roc_interp.csv'  # Interpolated ROC curve data (PerformanceData.roc_int)
ROC_NOCI = 'roc.csv'  # Non-interpolated ROC curve data (PerformanceData.roc)

PR_CI = 'pr_interp.csv'  # Interpolated PR curve data  (PerformanceData.pr_int)
PR_GAIN_CI = 'pr_gain.csv'  # Proportion reduction in positive tests and delta PPV from PR curve (PerformanceData.pr_gain)
PR_NOCI = 'pr.csv'  # Non-interpolated PR curve data (PerformanceData.pr)

DC_CI = 'dc.csv'  # Decision curve data (PerformanceData.dc)


# Output file names for reformatted discrimination, calibration, net benefit tables
DISC_PAPER = 'reformat_discrimination.csv'
CAL_PAPER = 'reformat_calibration.csv'
RISK_PAPER = 'reformat_metrics_at_risk.csv'
SENS_PAPER = 'reformat_metrics_at_sens.csv'
SENS_FIT_PAPER = 'reformat_metrics_at_sens_fit.csv'
SENS_FIT_PAPER2 = 'reformat_metrics_at_fit_and_mod_thresholds.csv'
DC_PAPER = 'reformat_dca.csv'
SENS_GAIN_PAPER = 'reformat_pr_gain.csv'