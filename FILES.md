## Overview of files 


### At a glance 

`./dataprep_fit` : contains code for the initial steps of processing the FIT data (extracting clinical symptoms and identifying tests done in the stool pot), and code for identifying pathology reports that discuss current colorectal cancer. These steps were run in the hospital information system as part of extracting all data items needed for analysis and pseudonymising them. The functions in the `./dataprep_fitval` directory were later run on pseudonymised data to further clean the data and transform it into a format where a prediction model can be applied.

`./dataprep_fitval` : contains functions that were used to prepare the hospital data for analysis (e.g. to clean the blood test values and extract the blood test value closest to the first FIT test), and to summarise changes in the patient population over time (e.g. to plot the proportion of patients with a positive FIT test over time). These scripts were originally contained in a separate code repository: the `./dataprep_fitval` directory thus includes its own requirements.txt file and installation instructions in README.md.

`./fitval` : contains helper functions needed for validating the COLOFIT prediction model, such as functions that compute performance metrics;

`./runs` : contains scripts that were run to validate the COLOFIT prediction model and produce the tables and figures in the publication (these scripts make use of the helper functions in the `./fitval` directory);

`./tests` : contains scripts that were used to check that the functions are performing as expected;


### In detail

`./dataprep_fit/runs/crc_from_path_20240617.py` : code for identifying pathology reports that discuss current colorectal cancer.

`./dataprep_fit/runs/fit_data_clean_20240612.py` : code for identifying FIT tests done in the stool pot and extracting clinical symptoms from the notes associated with the FIT test.

`./dataprep_fit/runs/tnm_staging_20240619.py` : code for extracting TNM staging from the clinical reports 

`./dataprep_fit/textmining` : contains scripts that help extract information from clinical reports. These are regex-based algorithms for identifying reports that discuss CRC, extracting explicitly given TNM staging scores, and extracting symptoms. The scripts are also documented in their own repository at https://github.com/tammandres/crc-and-tnm/tree/main/textmining .

`./dataprep_fitval/dataprep` : various helper functions for cleaning and preparing each data item. E.g. the 'bloods.py' script prepares the blood test values.

`./dataprep_fitval/runs` : scripts that were run to prepare the dataset and to analyse time trends in the FIT test values.

`./fitval/R/check-cal-in-r.R` : checks that calibration metrics computed by the current python pipeline are the same as those returned by different calibration packages in R.

`./fitval/R/dca.R` : computes decision curves in R, to check that these are the same as computed by the python package.

`./fitval/boot.py` : functions that apply the COLOFIT prediction model to a dataset and compute performance metrics with bootstrap confidence intervals

`./fitval/dummydata.py` : functions to create a simple synthetic dataset for testing the code

`./fitval/metrics.py` : functions that compute a variety of performance metrics given binary outcome labels (e.g. 0 - no cancer, 1 - cancer) and predicted risk scores

`./fitval/models.py` : implementation of the main COLOFIT Cox model and other COLOFIT models in python

`./fitval/modexplore.py` : functions to explore the COLOFIT model by visualising the contribution of each input variable to the linear predictor 

`./fitval/plots.py` : functions to plot performance metrics 

`./fitval/reformat.py` : functions that help reformat the tables of performance metrics (e.g. giving columns more descriptive names) so they can be more easily included in a publication

`./runs/airlock_copy_20240827.py` : an example script to copy some outputs to a separate directory so that they could be exported out of the secure data environment

`./runs/colofit_agg_reformat_descriptives.py` : reformat the descriptive statistics table that contains descriptives for all time periods

`./runs/colofit_agg_reformat_tables.py` : further formatting of aggregated performance metric tables

`./runs/colofit_agg.py` : combines the tables of performance metrics that were computed separately for each time period, into a single table including all time periods

`./runs/colofit_descriptives_alldata.py` : computes some descriptive statistics on the entire dataset (not separated by cancer status) for inclusion in the publication, such as the overall percentage of people in different ethnic groups

`./runs/colofit_hm_vs_oc.py` : explore how the predicted risk changes when FIT values are increased above 400 while other predictors are set at median. Provides insight into how predicted risks may differ for patients who did their FIT with HMJack rather than OC censor.

`./runs/colofit_performance_orig.py` : visualise COLOFIT performance in Nottingham and Oxford data over similar time periods

`./runs/colofit_reduction_external-local.py` : computes reduction in referrals for each time period, using different methods of computing the risk score threshold that captures the same number of cancers as FIT test at threshold >= 10. The methods are: estimating the threshold on Nottingham data; estimating the threshold locally (in Oxford data) on the current time period; estimating the threshold locally on the previous time period (to mimic real-world usage).

`./runs/colofit_reduction-by-predictor_external-local.py` : computes reduction in referrals for each time period under different methods of computing the risk score threshold, but this time also including different subsets of the linear predictor to evaluate how the predictor variables contribute to reduction in referrals over time.

`./runs/colofit_reformat.py` : reformats tables of performance metrics so that they are closer to a format required for publishing

`./runs/colofit_sample_proportion.py` : conducts statistical tests for comparing patients with and without cancer, e.g. comparing the proportion of patients with low haemoglobin

`./runs/colofit_sanitycheck.py` : a few checks, including a check that more manually calculated results give the same results as the scripts

`./runs/colofit_time-cut_descriptives_fix-bloods-high-low.py` : fixes a previous issue where the proportion of patients with low or normal haemoglobin was incorrectly calculated in some of the time periods

`./runs/colofit_time-cut_descriptives.py` : compute descriptive statistics for each time period in the dataset

`./runs/colofit_time-cut_metrics.py` : computes performance metrics separately for each time period in the dataset

`./runs/colofit_time-cut_pfit10.py` : compute proportion of patients with FIT >= 10 ug/g in each time period

`./runs/colofit_time-cut_plot-followup.py` : plot reduction in referrals with 180 and 365 day follow-up periods

`./runs/colofit_time-cut_plot.py` : plot perormance metrics across time periods in the dataset

`./runs/colofit_time-prepost_descriptives.py` : compute descriptive statistics before and after buffer device adoption

`./runs/colofit_time-prepost_metrics.py` : computes performance metrics before and after buffer device adoption

`./runs/colofit_time-prepost_plot.py` : plot performance metrics before and after buffer device adoption

`./runs/README_runs.md` : additionally describes the scripts in the order of which they were run