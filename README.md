# External validation of the COLOFIT colorectal cancer risk prediction model in the Oxford dataset

This repository accompanies the publication `External validation of the COLOFIT colorectal cancer risk prediction model in the Oxford-FIT dataset: the importance of population characteristics and clinically relevant evaluation metrics` authored by Andres Tamm, Brian Shine, Tim James, Jaimie Withers, Hizni Salih, Theresa Noble, Kinga A. Várnai, James E. East, Gary Abel, Willie Hamilton, Colin Rees, Eva JA Morris, Jim
Davies, and Brian D Nicholson; and submitted to BMC Medicine.

The repository serves two goals: 

(1) To contain a snapshot of the code that was used to produce the outputs included in the publication. Some of this code cannot be run externally, because the hospital data that it is meant to be run on cannot be shared. The snapshot can be found in the release "manuscript": https://github.com/tammandres/fitval/releases/tag/v1.0.0-manuscript

(2) To facilitate the evaluation of FIT-test based prediction models in the future. The current release contains a few improvements to the code, and examples of how this repository can be used to evaluate a prediction model on a new dataset.

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


## Overview of files 

`./dataprep_fit` : contains code for the initial steps of processing the FIT data (extracting clinical symptoms and identifying tests done in the stool pot), and code for identifying pathology reports that discuss current colorectal cancer. These steps were run in the hospital information system as part of extracting all data items needed for analysis and pseudonymising them. The functions in the `./dataprep_fitval` directory were later run on pseudonymised data to further clean the data and transform it into a format where a prediction model can be applied.

`./dataprep_fitval` : contains functions that were used to prepare the hospital data for analysis (e.g. to clean the blood test values and extract the blood test value closest to the first FIT test), and to summarise changes in the patient population over time (e.g. to plot the proportion of patients with a positive FIT test over time). These scripts were originally contained in a separate code repository: the `./dataprep_fitval` directory thus includes its own requirements.txt file and installation instructions in README.md.

`./fitval` : contains helper functions needed for validating the COLOFIT prediction model, such as functions that compute performance metrics;

`./runs` : contains scripts that were run to validate the COLOFIT prediction model and produce the tables and figures in the publication (these scripts make use of the helper functions in the `./fitval` directory);

`./tests` : contains scripts that were used to check that the functions are performing as expected;


See `FILES.md` for a more detailed description of the scripts included in this repository.
