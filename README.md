# External validation of the COLOFIT colorectal cancer risk prediction model in the Oxford dataset

This repository accompanies the publication `External validation of the COLOFIT colorectal cancer risk prediction model in the Oxford-FIT dataset: the importance of population characteristics and clinically relevant evaluation metrics` authored by Andres Tamm, Brian Shine, Tim James, Jaimie Withers, Hizni Salih, Theresa Noble, Kinga A. Várnai, James E. East, Gary Abel, Willie Hamilton, Colin Rees, Eva JA Morris, Jim
Davies, and Brian D Nicholson; and submitted to BMC Medicine.

The COLOFIT colorectal cancer risk prediction model uses the faecal immunochemical test (FIT) results, gender, age and two blood test results to produce a risk score for colorectal cancer. The model was developed in Nottingham, using data of patients who had symptoms of cancer and who were offered the FIT test by their GP.

This repository serves two goals: 

(1) To contain a snapshot of the code that was used to produce the outputs included in the publication. Some of this code cannot be run externally, because the hospital data that it is meant to be run on cannot be shared. The snapshot can be found in the release "Manuscript": https://github.com/tammandres/fitval/releases/tag/v1.0.0-manuscript  **NB**: the functions that were used to preprocess the hospital data are included in that release, but not in the most up-to-date version of this repository, because they are specific to the Oxford data and are not essential for validating a prediction model on a new dataset.

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

It is more efficient to run the code on a dataframe that already contains model predictions, as that data can be reused in each bootstrap sample. The only reason to apply prediction model to data in each bootstrap sample is when data has missing values. In that case, the data is imputed in each sample, and a model is applied to imputed datasets in each sample. This, however, will also be more time consuming.


## Overview of files 


**At a glance**: 

`./dataprep_fit` : contains code for the initial steps of processing the FIT data (extracting clinical symptoms and identifying tests done in the stool pot), and code for identifying pathology reports that discuss current colorectal cancer. Once these steps were complete, the functions in the `./dataprep_fitval` directory were used to prepare a dataset for analysis. This code will be uploaded within a week.

`./dataprep_fitval` : contains functions that were used to prepare the hospital data for analysis (e.g. to clean the blood test values and extract the blood test value closest to the first FIT test), and to summarise changes in the patient population over time (e.g. to plot the proportion of patients with a positive FIT test over time). NB - these scripts were originally used as a separate code repository. The `./dataprep_fitval` directory thus includes its own requirements.txt file and installation instructions in README.md.

`./fitval` : contains helper functions needed for the analysis, such as functions that compute performance metrics;

`./runs` : contains scripts that were run to conduct the analysis and produce the tables and figures in the publication (these scripts make use of the helper functions in the `./fitval` directory);

`./tests` : contains scripts that were used to check that the functions are performing as expected;
