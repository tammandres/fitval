# Validation of FIT-test based prediction models for colorectal cancer

This repository accompanies the publication `External validation of the COLOFIT colorectal cancer risk prediction model in the Oxford-FIT dataset: the importance of population characteristics and clinically relevant evaluation metrics` authored by Andres Tamm, Brian Shine, Tim James, Jaimie Withers, Hizni Salih, Theresa Noble, Kinga A. Várnai, James E. East, Gary Abel, Willie Hamilton, Colin Rees, Eva JA Morris, Jim
Davies, and Brian D Nicholson; and submitted to BMC Medicine. 

The COLOFIT colorectal cancer risk prediction model uses the faecal immunochemical test (FIT) results, gender, age and two blood test results to produce a risk score for colorectal cancer. The model was developed in Nottingham, using data of patients who had symptoms of cancer and who were offered the FIT test by their GP.

This repository serves two goals: 

(1) To contain a snapshot of the code that was used to produce the outputs included in the publication. Some of this code cannot be run externally, because the hospital data that it is meant to be run on cannot be shared. The snapshot can be found in the release "Manuscript": https://github.com/tammandres/fitval/tree/v0.01-manuscript  **NB**: the functions that were used to preprocess the hospital data are included in that release, but not in the most up-to-date version of this repository, because they are specific to the Oxford data and are not essential for validating a prediction model on a new dataset.

(2) To facilitate the evaluation of colorectal cancer risk prediction models in the future. The most up-to-date version of this repository contains some improvements to the code so that it would be easier to use on new data. It also contains examples of how the code can be used to evaluate a prediction model against the faecal immunochemical test (FIT) alone. Usually, the prediction model would include the FIT test and other variables like age, sex and blood test results, with the aim of improving on the FIT test alone. A key performance metric is the 'reduction in referrals': the model would improve on the FIT test alone, if it would capture the same number of colorectal cancers as FIT, but lead to less patients testing positive compared to FIT. In that case, less patients would need to be offered further investigations (usually colonoscopy) to detect the same number of cancers. The reduction in referrals is computed as a percentage: if it is negative (e.g. -20%), it shows that potentially 20% less patients would need to be investigated to capture the same number of cancers as FIT alone.

The code is currently published using the CC BY-NC license (which allows sharing and adapting but not commercial use - https://creativecommons.org/licenses/by-nc/4.0). This is because the repository also contains an implementation of the COLOFIT risk prediction model.


## Installation

Install Python packages under requirements:

```
conda create -n fitval python=3.9
conda activate fitval
cd <path-to-code-dir>  # e.g. cd C:\\Users\\2lnM\\Desktop\\project_fitml\\fitval
pip install -r requirements.txt
pip install -e .
```

## Usage

See `./examples/run.py`.

The code can be run on a table that contains the colorectal cancer indicator, FIT test values, and predicted probabilities of CRC from a prediction model for each patient. The columns must be named and formatted as follows:

* `y_true` : indicator variable for colorectal cancer (0: no, 1: yes), 

* `fit_val` : numerical FIT test values (1.5, 10, 20, etc), 

* `y_pred` : predicted probability of CRC according to a model, must be in the [0, 1] interval (0.03, 0.05, 0.25 etc). Multiple models can also be included, in which case the columns must be named as y_pred1, y_pred2, ... etc.


The table must be in a `.csv` format. For example, if the table is located in `./data/predictions.csv` (which currently contains a simple synthetic dataset), then the code can be run as follows.

```python
from fitval.boot import boot_metrics
from fitval.plots import plot_cal_smooth, plot_rocpr_interp, plot_dca, plot_cal_bin
from constants import PROJECT_ROOT
data_path = PROJECT_ROOT / 'data' / 'predictions.csv'
save_path = PROJECT_ROOT / 'results'
save_path.mkdir(exist_ok=True, parents=True)

# Compute all performance metrics over 100 bootstrap samples
data_ci, data_noci = boot_metrics(data_path=data_path, save_path=save_path, model_names=['fit-age-sex-bloods', 'fit-age-sex'], 
                                  data_has_predictions=True, B=100, parallel=False, nchunks=15, plot_boot=False, recal=False)

# Plot performance metrics 
plot_cal_smooth(save_path)  # plot smooth calibration curves based on LOWESS smoothing
plot_rocpr_interp(save_path)  # plot interpolated ROC and precision-recall curves
plot_dca(save_path)  # plot decision curves
plot_cal_bin(save_path)  # plot calibration curves based on dividing data into bins

# If the aim is to only compute reduction in referrals relative to the FIT test, 
metrics, metrics_reformat = boot_reduction_in_referrals(data_path, save_path, model_names=['fit-age-sex-bloods', 'fit-age-sex'],
                                                        thr_fit = [10], B=100, plot_boot=True)

# see ./examples/run.py for more comments
```

The code can also be run on data with missing values, in which case the `get_model` function needs to be specified in `./fitval/models.py`. In that case, the data will be imputed within in each bootstrap sample and the prediction model applied to the bootstrap sample. Alternatively, a suitable missing data model could be trained on a subset of data that is not used for evaluating the model, and then applied to a held-out subset which will be used for evaluating the model. This would mimic real-world usage in a missing data scenario, and the held-out subset could be used for evaluating the imputer-model combination in the same way as illustrated above.


## Overview of files 

`./data` : a simple synthetic dataset to illustrate how this repository can be used;

`./examples` : scripts that illustrate how the code can be run on new data (in these scripts, the code is run on synthetic data to illustrate how it works);

`./fitval` : helper functions needed for evaluating a prediction model, such as functions that compute performance metrics;

`./tests` : scripts that were used to check that the functions are performing as expected;
